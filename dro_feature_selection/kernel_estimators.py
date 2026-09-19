"""Differentiable kernel estimators used by the feature-selection objective."""

import torch
from sklearn.model_selection import KFold
from torch import Tensor

from global_vars import CLAMP_MAX_ALPHA, CLAMP_MIN_ALPHA


def estimate_conditional_expectation_knn(
        X_ref: torch.Tensor,       # Reference features [n_ref, d]
        S_query: torch.Tensor,     # Query features S(alpha) [n_query, d]
        E_Y_X_ref: torch.Tensor,   # Reference E[Y|X] values [n_ref]
        alpha: torch.Tensor,       # Noise parameters [d]
        k: int = 1000,             # Number of neighbors
        clamp_min: float = 1e-5,   # Min value for alpha and squared distances
        clamp_max_dist: float = 1e6 # Max value for squared distances
        ) -> torch.Tensor:
    """
    Differentiable kNN kernel-weighted estimate of E[Y|S_query] using references.
      W_ij ~ exp( -(1/2)* || (S_i - X_j) / sqrt(alpha) ||^2 )
      E[Y|S_i] = sum_{j in kNN(S_i)} W_ij * E_Y_X_ref[j].

    Ensures tensors are on the same device as S_query.
    """
    device = S_query.device
    n_ref = X_ref.shape[0]
    n_query = S_query.shape[0]

    # Move reference data to the correct device if necessary
    X_ref = X_ref.to(device)
    E_Y_X_ref = E_Y_X_ref.to(device)
    alpha = alpha.to(device) # Ensure alpha is also on the right device

    # 1) Clamp alpha and compute inverse sqrt variance
    alpha_safe = torch.clamp(alpha, min=clamp_min)       # (d,)
    inv_sqrt_alpha = torch.rsqrt(alpha_safe)             # (d,) -> 1/sqrt(alpha)

    # 2) Scale features into Mahalanobis space based on alpha
    Xs_scaled = X_ref * inv_sqrt_alpha                   # (n_ref, d)
    Ss_scaled = S_query * inv_sqrt_alpha                 # (n_query, d)

    # 3) Compute pairwise squared distances in scaled space
    #    D2_ij = || Ss_scaled_i - Xs_scaled_j ||^2
    #    Using cdist is generally efficient and stable
    if n_ref > 10000:
        # Compute squared distances in mini-batches to reduce peak memory
        batch_size = 50
        D2_chunks = []
        for i in range(0, n_query, batch_size):
            chunk = Ss_scaled[i:i+batch_size]
            D2_chunks.append(torch.cdist(chunk, Xs_scaled, p=2).pow(2))
        D2 = torch.cat(D2_chunks, dim=0)
    else:
        D2 = torch.cdist(Ss_scaled, Xs_scaled, p=2).pow(2)     # (n_query, n_ref)

    # Clamp distances to avoid potential numerical issues (optional but safe)
    D2 = torch.clamp(D2, min=clamp_min, max=clamp_max_dist) # (n_query, n_ref)

    # 4) Find k-nearest neighbors for each query point S_i based on scaled distance
    actual_k = min(k, n_ref)
    if actual_k < 1:
        print(f"Warning: actual_k={actual_k} < 1 in kernel estimation. Returning mean.")
        # Return mean of reference E[Y|X] as fallback
        return torch.full((n_query,), E_Y_X_ref.mean(), device=device, dtype=S_query.dtype)

    # topk finds the k smallest distances and their indices
    # Use torch.no_grad() for idx finding if not backpropping through indices (usually safe)
    with torch.no_grad():
         # D2_knn: distances to k nearest neighbors (n_query, k)
         # knn_indices: indices of these neighbors in X_ref (n_query, k)
        D2_knn, knn_indices = torch.topk(D2, actual_k, dim=1, largest=False)

    # Important: Re-select distances using indices *within* the computation graph
    # if gradients through D2 are needed for alpha (which they are).
    # Gather the distances corresponding to the selected indices.
    # This ensures the gradient path for D2 -> alpha is maintained.
    D2_knn_grad = D2.gather(1, knn_indices) # (n_query, k)

    # 5) Calculate weights using softmax over the k neighbors
    #    logW = -0.5 * D2_knn (use the version with grad)
    logW = -0.5 * D2_knn_grad                            # (n_query, k)
    W = torch.softmax(logW, dim=1)                       # (n_query, k), rows sum to 1

    # 6) Gather the E[Y|X] values for the k neighbors
    #    knn_indices shape: (n_query, k)
    #    E_Y_X_ref shape: (n_ref,) -> Need to index E_Y_X_ref using knn_indices
    #    Use gather or direct indexing
    E_Y_X_neighbors = E_Y_X_ref[knn_indices]             # (n_query, k)

    # 7) Compute weighted average
    E_Y_S_estimate = (W * E_Y_X_neighbors).sum(dim=1)    # (n_query,)

    return E_Y_S_estimate


def estimate_conditional_kernel_oof(
    X_batch: torch.Tensor,
    S_batch: torch.Tensor,
    E_Y_X: torch.Tensor,
    alpha: torch.Tensor,
    n_folds: int = 5,
    clamp_min: float = 1e-4,
    clamp_max: float = 1e6,
    k: int = 100,
    seed: int = 42
) -> torch.Tensor:
    """
    Out-of-fold kNN-kernel estimates for E[Y|S].
    """
    n_test = S_batch.size(0)
    oof = torch.zeros(n_test, device=X_batch.device)

    # Ensure k is not larger than the smallest possible training fold size
    min_train_size = max(1, int(X_batch.shape[0] * (1 - 1/n_folds)) if n_folds > 1 else X_batch.shape[0])
    actual_k = min(k, min_train_size)
    if actual_k < 1:
        print("Warning: k adjusted to 0 in estimate_conditional_kernel_oof. Returning zeros.")
        return oof # Or handle differently

    if n_folds <= 1:
        oof = estimate_conditional_expectation_knn(
            X_batch, S_batch, E_Y_X, alpha, k=actual_k, clamp_min=clamp_min, clamp_max=clamp_max
        )
        return oof
    else:
        # Note: This OOF implementation for the kernel estimator is slightly different
        # from the plugin/IF OOF. Here, for each test fold of S_batch, it uses the
        # *entire* X_batch and E_Y_X as the "training" set for the kernel weighting.
        # This might be intended, but differs from typical CV where the model/reference
        # data is also split. If true OOF is needed, X_batch and E_Y_X should also be split.
        # Assuming current implementation is intended:
        kf  = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for _, te_idx in kf.split(S_batch): # We only need test indices for S_batch
            oof[te_idx] = estimate_conditional_expectation_knn(
                X_batch, S_batch[te_idx], E_Y_X, alpha, k=actual_k, clamp_min=clamp_min, clamp_max=clamp_max
            )
        return oof


def estimate_conditional_keops(
    X: Tensor,           # (n_train, d) - Reference features
    S: Tensor,           # (n_test, d)  - Query features
    E_Y_X: Tensor,       # (n_train,) - Reference values
    alpha: Tensor,       # (d,)       - Noise parameters
    k: int = 1000,       # Number of neighbors to use
    clamp_min: float = 1e-4, # Min clamping for alpha variance
    # clamp_max_dist: float = 1e6 # Optional: Max clamping for squared distances
) -> Tensor:
    """
    Estimates E[ E_Y_X | S=s ] using PyTorch k-NN kernel weighting
    on variance-scaled features. Maintains differentiability w.r.t. alpha.

    Args:
        X: Reference features tensor.
        S: Query features tensor.
        E_Y_X: Reference values tensor (E[Y|X]).
        alpha: Noise variance parameters tensor.
        k: Number of nearest neighbors to consider.
        clamp_min: Minimum value for variance clamping.
        # clamp_max_dist: Optional max value for squared distances.

    Returns:
        Tensor: Estimated values E[Y|S] for each query point in S.
    """
    device = S.device
    n_train = X.shape[0]
    n_test = S.shape[0]
    d = X.shape[1]

    # --- Input Validation ---
    if E_Y_X.ndim != 1 or E_Y_X.shape[0] != n_train:
        raise ValueError(f"E_Y_X must be a 1D tensor of shape ({n_train},), but got shape {E_Y_X.shape}")
    if alpha.shape != (d,):
         raise ValueError(f"alpha must be a 1D tensor of shape ({d},), but got shape {alpha.shape}")
    if X.device != device or E_Y_X.device != device or alpha.device != device:
        # For simplicity in this example, move all to S's device.
        # In practice, ensure inputs are on the desired device beforehand.
        X = X.to(device)
        E_Y_X = E_Y_X.to(device)
        alpha = alpha.to(device)
        # print("Warning: Moving input tensors to device:", device)


    # --- 1. Scaling ---
    # Clamp alpha (variance) and compute inverse square root
    alpha_clamped = torch.clamp(alpha, min=clamp_min)       # (d,)
    inv_sqrt_var_t = torch.rsqrt(alpha_clamped)             # (d,) -> 1/sqrt(variance)

    # Scale features into Mahalanobis-like space
    # Ensure broadcasting works correctly: (n, d) * (d,) -> (n, d)
    Xs_scaled = X * inv_sqrt_var_t                   # (n_train, d)
    Ss_scaled = S * inv_sqrt_var_t                   # (n_test, d)

    # --- 2. Pairwise Distances ---
    # Calculate squared Euclidean distances in the scaled space
    # This operation is differentiable w.r.t. Ss_scaled, Xs_scaled, and thus alpha
    if n_train > 10000:
        # Compute squared distances in mini-batches to reduce peak memory
        batch_size = 50
        D2_chunks = []
        for i in range(0, n_test, batch_size):
            chunk = Ss_scaled[i:i+batch_size]
            D2_chunks.append(torch.cdist(chunk, Xs_scaled, p=2).pow(2))
        D2 = torch.cat(D2_chunks, dim=0) # (n_test, n_train)
    else:
        D2 = torch.cdist(Ss_scaled, Xs_scaled, p=2).pow(2)     # (n_test, n_train)

    # Optional: Clamp distances for numerical stability
    # D2 = torch.clamp(D2, min=clamp_min, max=clamp_max_dist)

    # --- 3. Find K-Nearest Neighbors ---
    # Determine the actual k to use (cannot exceed n_train)
    actual_k = min(k, n_train)
    if actual_k < 1:
        print(f"Warning: actual_k={actual_k} < 1. Returning mean E[Y|X].")
        # Fallback: return the mean of reference values
        return torch.full((n_test,), E_Y_X.mean(), device=device, dtype=S.dtype)

    # Find the indices and *squared distances* of the k nearest neighbors
    # `topk` returns (values, indices). values are the smallest squared distances here.
    # Importantly, the returned distances (D2_knn) retain gradient history.
    D2_knn, knn_indices = torch.topk(D2, actual_k, dim=1, largest=False) # (n_test, k)

    # --- 4. Calculate Weights ---
    # Gaussian kernel weights based on the squared distances of the k neighbors
    logW = -0.5 * D2_knn                            # (n_test, k)
    # Normalize weights using softmax across the k neighbors for each test point
    W = torch.softmax(logW, dim=1)                  # (n_test, k), rows sum to 1

    # --- 5. Weighted Sum ---
    # Gather the E[Y|X] values for the k neighbors using the indices
    # E_Y_X[knn_indices] uses advanced indexing and works correctly
    E_Y_X_neighbors = E_Y_X[knn_indices]             # (n_test, k)

    # Compute the weighted average
    E_Y_S_estimate = (W * E_Y_X_neighbors).sum(dim=1)    # (n_test,)

    return E_Y_S_estimate


def chunked_pairwise_distance(X: torch.Tensor, S: torch.Tensor, chunk_size: int = 50) -> torch.Tensor:
    """
    Calculates pairwise distances between X and S in a memory-efficient way using chunks.
    Preserves gradient flow while reducing peak memory usage.
    
    Args:
        X: First set of points (n_train, d)
        S: Second set of points (n_test, d)
        chunk_size: Size of chunks to process at a time
        
    Returns:
        Pairwise squared distances (n_test, n_train)
    """
    n_test, d = S.shape
    n_train = X.shape[0]
    device = S.device
    
    # Pre-allocate the full distance matrix
    D2 = torch.zeros((n_test, n_train), device=device)
    
    # Process in chunks to reduce memory usage
    for i in range(0, n_test, chunk_size):
        end_idx = min(i + chunk_size, n_test)
        chunk = S[i:end_idx]
        
        # Calculate distances for this chunk
        try:
            D2[i:end_idx] = torch.cdist(chunk, X, p=2).pow(2)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                # Fallback to even smaller chunks if necessary
                for j in range(i, end_idx):
                    D2[j:j+1] = torch.cdist(S[j:j+1], X, p=2).pow(2)
            else:
                raise e
                
    return D2


def estimate_conditional_keops_flexible_optimized(
    X: torch.Tensor,           # (n_train, d) - Reference features
    S: torch.Tensor,           # (n_test, d)  - Query features
    E_Y_X: torch.Tensor,       # (n_train,) - Reference values
    param: torch.Tensor,       # (d,)       - Either alpha or theta
    param_type: str = 'alpha', # 'alpha' or 'theta'
    k: int = 1000,
    clamp_min_alpha: float = 1e-4, 
    max_batch_size: int = 500  # Max batch size to control memory usage
) -> torch.Tensor:
    """
    Memory-efficient GPU implementation that avoids OOM errors
    by processing in smaller batches while staying on GPU.
    """
    device = param.device
    n_train, d = X.shape
    n_test = S.shape[0]
    
    # Scale features according to parameter type
    if param_type == 'alpha':
        alpha_clamped = torch.clamp(param, min=clamp_min_alpha)
        inv_sqrt_var_t = torch.rsqrt(alpha_clamped)
    elif param_type == 'theta':
        inv_sqrt_var_t = torch.exp(-param / 2.0)
    else:
        raise ValueError("param_type must be 'alpha' or 'theta'")
    
    # Scale reference points (only once)
    Xs_scaled = X * inv_sqrt_var_t
    
    # Create output tensor
    results = torch.zeros(n_test, dtype=S.dtype, device=device)
    
    # Automatically determine batch size based on data dimensions
    # Adjust the divisor based on your GPU memory
    min_batch_size = 1
    suggested_batch_size = max(min_batch_size, min(max_batch_size, int(1e9 / (n_train * d * 4))))
    batch_size = suggested_batch_size
    
    # Process in batches to manage memory
    for i in range(0, n_test, batch_size):
        end_i = min(i + batch_size, n_test)
        batch_S = S[i:end_i]
        
        # Scale query points for this batch
        batch_S_scaled = batch_S * inv_sqrt_var_t
        
        # Compute distances efficiently in chunks if reference set is large
        if n_train > 10000:
            inner_batch_size = min(5000, n_train)
            D2 = torch.zeros((end_i - i, n_train), device=device)
            
            for j in range(0, n_train, inner_batch_size):
                end_j = min(j + inner_batch_size, n_train)
                # Compute and store partial distance matrix
                D2[:, j:end_j] = torch.cdist(batch_S_scaled, Xs_scaled[j:end_j], p=2).pow(2)
                
                # Free memory explicitly
                torch.cuda.empty_cache()
        else:
            # If reference set is small enough, compute in one go
            D2 = torch.cdist(batch_S_scaled, Xs_scaled, p=2).pow(2)
        
        # Find k nearest neighbors
        k_actual = min(k, n_train)
        dists, indices = torch.topk(D2, k_actual, largest=False, dim=1)
        
        # Free the distance matrix explicitly
        del D2
        torch.cuda.empty_cache()
        
        # Compute weights
        weights = torch.exp(-0.5 * dists)
        weights_sum = torch.sum(weights, dim=1, keepdim=True)
        weights = weights / torch.clamp(weights_sum, min=1e-8)
        
        # Get values and compute weighted sum
        values = E_Y_X[indices]
        results[i:end_i] = torch.sum(weights * values, dim=1)
        
        # Free batch memory
        del batch_S_scaled, dists, indices, weights, values
        torch.cuda.empty_cache()
        
    return results


def estimate_conditional_keops_flexible(
    X: torch.Tensor,           # (n_train, d) - Reference features
    S: torch.Tensor,           # (n_test, d)  - Query features
    E_Y_X: torch.Tensor,       # (n_train,) - Reference values
    param: torch.Tensor,       # (d,)       - Either alpha or theta
    param_type: str = 'alpha', # 'alpha' or 'theta'
    k: int = 1000,
    clamp_min_alpha: float = 1e-4, # Min clamping for alpha variance
    chunk_size: int = 50       # Size of chunks for distance calculation
) -> torch.Tensor:
    """
    Estimates E[ E_Y_X | S=s ] using PyTorch k-NN kernel weighting.
    Uses chunked distance calculation to reduce memory usage while preserving gradients.
    
    Args:
        X: Reference features tensor
        S: Query features tensor
        E_Y_X: Reference values tensor (E[Y|X])
        param: Noise variance parameters tensor (alpha or theta)
        param_type: 'alpha' or 'theta'
        k: Number of nearest neighbors to consider
        clamp_min_alpha: Minimum value for alpha clamping
        chunk_size: Size of chunks for distance calculation
        
    Returns:
        Tensor: Estimated values E[Y|S] for each query point in S
    """
    # Store original device
    device = param.device
    
    # Get tensor dimensions
    n_train, d = X.shape
    n_test = S.shape[0]
    
    # --- Input Validation ---
    if E_Y_X.ndim != 1 or E_Y_X.shape[0] != n_train:
        raise ValueError(f"E_Y_X shape mismatch: expected ({n_train},), got {E_Y_X.shape}")
    if param.shape != (d,):
         raise ValueError(f"Parameter shape mismatch: expected ({d},), got {param.shape}")
    
    # Ensure inputs are on the same device
    X, E_Y_X = X.to(device), E_Y_X.to(device)
    
    # --- 1. Scaling based on param_type ---
    if param_type == 'alpha':
        alpha_clamped = torch.clamp(param, min=clamp_min_alpha)
        inv_sqrt_var_t = torch.rsqrt(alpha_clamped) # 1/sqrt(alpha)
    elif param_type == 'theta':
        inv_sqrt_var_t = torch.exp(-param / 2.0) # 1/sqrt(exp(theta))
    else:
        raise ValueError("param_type must be 'alpha' or 'theta'")

    # Apply scaling - this preserves gradients
    Xs_scaled = X * inv_sqrt_var_t
    Ss_scaled = S * inv_sqrt_var_t
    
    # Actual k to use (cannot exceed n_train)
    actual_k = min(k, n_train)
    if actual_k < 1:
        print(f"Warning: actual_k={actual_k} < 1. Returning mean E[Y|X].")
        return torch.full((n_test,), E_Y_X.mean(), device=device, dtype=S.dtype)
    
    # --- 2. Memory-efficient distance calculation ---
    try:
        # Try using chunked distance calculation
        D2 = chunked_pairwise_distance(Xs_scaled, Ss_scaled, chunk_size=chunk_size)
        
        # --- 3. Find K-Nearest Neighbors ---
        dists, inds = torch.topk(D2, actual_k, largest=False, dim=1)
        
        # --- 4. Calculate Weights ---
        W = torch.exp(-0.5 * dists)
        W = W / (torch.sum(W, dim=1, keepdim=True) + 1e-8)
        
        # --- 5. Weighted Sum ---
        E_Y_X_neighbors = E_Y_X[inds]
        E_Y_S_estimate = torch.sum(W * E_Y_X_neighbors, dim=1)
        
        return E_Y_S_estimate
        
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print(f"Warning: CUDA OOM with chunk size {chunk_size}. Processing one sample at a time.")
            
            # Process one sample at a time as a last resort
            results = []
            for i in range(n_test):
                # Calculate distances for one sample
                sample = Ss_scaled[i:i+1]
                try:
                    # Try on current device
                    D2_i = torch.cdist(sample, Xs_scaled, p=2).pow(2)
                    dists_i, inds_i = torch.topk(D2_i, actual_k, largest=False, dim=1)
                    W_i = torch.exp(-0.5 * dists_i)
                    W_i = W_i / (torch.sum(W_i, dim=1, keepdim=True) + 1e-8)
                    E_Y_X_neighbors_i = E_Y_X[inds_i]
                    result_i = torch.sum(W_i * E_Y_X_neighbors_i, dim=1)
                    results.append(result_i)
                    
                except RuntimeError:
                    # Last resort: move to CPU while preserving gradient flow
                    cpu_sample = sample.cpu()
                    cpu_Xs = Xs_scaled.cpu()
                    cpu_E_Y_X = E_Y_X.cpu()
                    
                    D2_i = torch.cdist(cpu_sample, cpu_Xs, p=2).pow(2)
                    dists_i, inds_i = torch.topk(D2_i, actual_k, largest=False, dim=1)
                    W_i = torch.exp(-0.5 * dists_i)
                    W_i = W_i / (torch.sum(W_i, dim=1, keepdim=True) + 1e-8)
                    E_Y_X_neighbors_i = cpu_E_Y_X[inds_i]
                    result_i = torch.sum(W_i * E_Y_X_neighbors_i, dim=1)
                    results.append(result_i.to(device))  # Back to original device
            
            # Combine results while preserving gradient flow
            return torch.cat(results, dim=0)
        else:
            # Re-raise if not OOM error
            raise e


def estimate_T2_mc_flexible(
    X_std_torch: torch.Tensor,
    E_Yx_std_torch: torch.Tensor,
    param_torch: torch.Tensor,
    param_type: str,
    n_mc_samples: int,
    k_kernel: int,
    chunk_size: int = 50      # Size of chunks for distance calculation
) -> torch.Tensor:
    """
    Estimates T2 = E[E[Y_std|S]^2] using Monte Carlo sampling and memory-efficient kernel estimator.
    Differentiable w.r.t. param_torch.
    
    Args:
        X_std_torch: Standardized features (n, d)
        E_Yx_std_torch: Standardized conditional expectations E[Y|X] (n,)
        param_torch: Parameter tensor (alpha or theta) (d,)
        param_type: 'alpha' or 'theta'
        n_mc_samples: Number of Monte Carlo samples
        k_kernel: Number of neighbors for kernel estimation
        chunk_size: Size of chunks for distance calculation
        
    Returns:
        Scalar estimate of T2 = E[E[Y_std|S]^2]
    """
    device = param_torch.device
    avg_term2_std = torch.zeros(1, device=device, dtype=torch.float32)

    # Move input tensors to GPU instead of moving param to CPU
    X_std_torch = X_std_torch.to(device)
    E_Yx_std_torch = E_Yx_std_torch.to(device)

    # # print the devices of all tensors
    # print(f"X_std_torch device: {X_std_torch.device}")
    # print(f"E_Yx_std_torch device: {E_Yx_std_torch.device}")
    # print(f"param_torch device: {param_torch.device}")

    
    # Determine noise scale based on param type
    if param_type == 'alpha':
        alpha_clamped = param_torch.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
        noise_scale = torch.sqrt(alpha_clamped)
        param_for_kernel = param_torch
    elif param_type == 'theta':
        alpha_derived = torch.exp(param_torch)
        alpha_clamped = alpha_derived.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
        noise_scale = torch.sqrt(alpha_clamped)
        param_for_kernel = param_torch
    else:
        raise ValueError("param_type must be 'alpha' or 'theta'")
    
    # Use dynamic MC sample count based on data size
    n_train = X_std_torch.shape[0]
    actual_mc_samples = min(n_mc_samples, 5 if n_train > 10000 else n_mc_samples)
    
    # Monte Carlo estimation
    for i in range(actual_mc_samples):
        epsilon_k = torch.randn_like(X_std_torch)
        S_param_k = X_std_torch + epsilon_k * noise_scale
        
        try:
            # Use memory-efficient kernel estimator
            E_Y_S_std_k = estimate_conditional_keops_flexible(
            X_std_torch, S_param_k, E_Yx_std_torch, 
            param_for_kernel, param_type, k=k_kernel,
            chunk_size=chunk_size
            )
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print("Warning: CUDA out of memory error detected. Switching to optimized estimator.")
                E_Y_S_std_k = estimate_conditional_keops_flexible_optimized(
                    X_std_torch, S_param_k, E_Yx_std_torch,
                    param_for_kernel, param_type, k=k_kernel,
                    max_batch_size=500
                )
            else:
                raise e
        
        term2_sample_std_k = E_Y_S_std_k.pow(2).mean()
        avg_term2_std += term2_sample_std_k
    
    return avg_term2_std / actual_mc_samples


def estimate_T2_kernel_IF_like_flexible(
    X_std_torch: torch.Tensor,
    Y_std_torch: torch.Tensor,
    E_Yx_std_torch: torch.Tensor,
    param_torch: torch.Tensor,
    param_type: str,
    n_mc_samples: int,
    k_kernel: int,
    chunk_size: int = 50      # Size of chunks for distance calculation
) -> torch.Tensor:
    """
    Estimates T2 using IF-like form: E[2*Y*mu_S - mu_S^2] with memory-efficient kernel estimator.
    Differentiable w.r.t. param_torch through all paths.
    
    Args:
        X_std_torch: Standardized features (n, d)
        Y_std_torch: Standardized outcomes (n,)
        E_Yx_std_torch: Standardized conditional expectations E[Y|X] (n,)
        param_torch: Parameter tensor (alpha or theta) (d,)
        param_type: 'alpha' or 'theta'
        n_mc_samples: Number of Monte Carlo samples
        k_kernel: Number of neighbors for kernel estimation
        chunk_size: Size of chunks for distance calculation
        
    Returns:
        Scalar estimate of T2 using IF-like form
    """
    device = param_torch.device
    avg_term2_if_like = torch.tensor(0.0, device=device, dtype=param_torch.dtype)
    
    # Determine noise scale based on param type
    if param_type == 'alpha':
        alpha_for_noise = param_torch.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
    elif param_type == 'theta':
        alpha_derived = torch.exp(param_torch)
        alpha_for_noise = alpha_derived.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
    else:
        raise ValueError("param_type must be 'alpha' or 'theta'")
    
    noise_scale = torch.sqrt(alpha_for_noise)
    
    # Use dynamic MC sample count based on data size
    n_train = X_std_torch.shape[0]
    actual_mc_samples = min(n_mc_samples, 5 if n_train > 10000 else n_mc_samples)
    
    # Monte Carlo estimation
    for i in range(actual_mc_samples):
        epsilon_k = torch.randn_like(X_std_torch)
        S_param_k = X_std_torch + epsilon_k * noise_scale
        
        # Use memory-efficient kernel estimator
        try:
            # Use memory-efficient kernel estimator
            mu_S_hat_k = estimate_conditional_keops_flexible(
                X_std_torch, S_param_k, E_Yx_std_torch,
                param_torch, param_type, k=k_kernel,
                chunk_size=chunk_size
            )
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print("Warning: CUDA out of memory error detected. Switching to optimized estimator.")
                mu_S_hat_k = estimate_conditional_keops_flexible_optimized(
                    X_std_torch, S_param_k, E_Yx_std_torch,
                    param_torch, param_type, k=k_kernel,
                    max_batch_size=500
                )
            else:
                raise e
        
        term2_sample_if_like_k = (2 * Y_std_torch * mu_S_hat_k - mu_S_hat_k.pow(2)).mean()
        avg_term2_if_like += term2_sample_if_like_k
    
    return avg_term2_if_like / actual_mc_samples


def estimate_E_Y_S_kernel_flexible(X_std_torch: torch.Tensor,
                                E_Yx_std_torch: torch.Tensor,
                                param_torch: torch.Tensor, # Detached alpha or theta
                                param_type: str,
                                n_mc_samples_S: int = 1,
                                k_kernel: int = 1000) -> torch.Tensor:
    """
    Estimates E[Y_std|S] value using the flexible kernel method.
    Averages over n_mc_samples_S realizations of S. Uses detached param.
    """
    avg_E_Y_S = torch.zeros(X_std_torch.shape[0], device=X_std_torch.device, dtype=X_std_torch.dtype)
    param_val = param_torch.detach().clone() # Use detached value

    if param_type == 'alpha':
        alpha_clamped = param_val.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
        noise_scale = torch.sqrt(alpha_clamped)
    elif param_type == 'theta':
        # Clamp the resulting alpha for noise stability
        alpha_for_noise = torch.exp(param_val).clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
        noise_scale = torch.sqrt(alpha_for_noise)
    else:
        raise ValueError("param_type must be 'alpha' or 'theta'")

    with torch.no_grad():
        for _ in range(n_mc_samples_S):
            epsilon_k = torch.randn_like(X_std_torch)
            S_param_k = X_std_torch + epsilon_k * noise_scale
            # Pass detached param for value estimation
        try:
            E_Y_S_std_k = estimate_conditional_keops_flexible(
                X_std_torch, S_param_k, E_Yx_std_torch, param_val, param_type, k=k_kernel
            )
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print("Warning: CUDA out of memory error detected. Switching to optimized estimator.")
                E_Y_S_std_k = estimate_conditional_keops_flexible_optimized(
                    X_std_torch, S_param_k, E_Yx_std_torch, param_val, param_type, k=k_kernel,
                    max_batch_size=500
                )
            else:
                raise e
        avg_E_Y_S += E_Y_S_std_k

    return avg_E_Y_S / n_mc_samples_S
