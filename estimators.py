# estimators.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree
from sklearn.base import clone # Import clone
import torch # Set number of threads for PyTorch
from torch import Tensor
import matplotlib.pyplot as plt
import xgboost as xgb # Import XGBoost
from typing import Optional
import math

from global_vars import *
torch.set_num_threads(CPU_COUNT)

def compute_penalty(alpha: torch.Tensor, # Input is always alpha
                    penalty_type: Optional[str],
                    penalty_lambda: float,
                    epsilon: float = EPS) -> torch.Tensor:
    """
    Compute a penalty term P(alpha) designed to encourage large alpha values.
    We minimize L = (T1 - T2) + P(alpha).
    """
    # Clamp alpha within the function for calculation, ensuring gradients flow
    alpha_clamped = torch.clamp(alpha, min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)

    if penalty_type is None or penalty_lambda == 0 or penalty_type.lower() == "none":
        return torch.tensor(0.0, device=alpha.device, dtype=alpha.dtype, requires_grad=alpha.requires_grad)

    penalty_type_lower = penalty_type.lower()

    if penalty_type_lower == "reciprocal_l1":
        return penalty_lambda * torch.sum(1.0 / (alpha_clamped + epsilon))
    elif penalty_type_lower == "neg_l1":
        print("Warning: Using Neg_L1 penalty encourages small alpha.")
        return penalty_lambda * torch.sum(torch.abs(alpha_clamped))
    elif penalty_type_lower == "max_dev":
        target_val = torch.tensor(1.0, device=alpha.device) # Target alpha=1
        return penalty_lambda * torch.sum(torch.abs(target_val - alpha_clamped))
    elif penalty_type_lower == "quadratic_barrier":
        return penalty_lambda * torch.sum((alpha_clamped + epsilon) ** (-2))
    elif penalty_type_lower == "exponential":
        return penalty_lambda * torch.sum(torch.exp(-alpha_clamped))
    else:
        raise ValueError("Unknown penalty_type: " + str(penalty_type))


# =============================================================================
# K-fold based estimators for conditional means and squared functionals
# =============================================================================

def plugin_estimator_conditional_mean(X, Y, estimator_type="rf", n_folds=N_FOLDS,
                                      seed=42):
    """
    Compute out-of-fold plugin predictions for E[Y|X] using K-fold CV.
    Supports 'rf', 'krr', and 'xgb' estimator types.
    """
    n_samples = X.shape[0]
    out_preds = np.zeros(n_samples)

    # Define model based on type
    if estimator_type == "rf":
        model_base = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    elif estimator_type == "krr":
        model_base = KernelRidge(kernel='rbf', alpha=0.1) # Example parameters
    elif estimator_type == "xgb":
        model_base = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=seed, n_jobs=-1, tree_method='hist') # Added XGBoost
    else:
        raise ValueError(f"Unsupported estimator_type: {estimator_type}. Choose 'rf', 'krr', or 'xgb'.")

    if n_folds <= 1:
        model = clone(model_base) # Use clone for fresh model
        model.fit(X, Y)
        return model.predict(X)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train = Y[train_idx]

            model = clone(model_base) # Use clone for fresh model per fold
            model.fit(X_train, Y_train)
            out_preds[test_idx] = model.predict(X_test)
        return out_preds

def plugin_estimator_squared_conditional(X, Y, estimator_type="rf", n_folds=N_FOLDS,
                                         seed=42):
    """
    Compute the plugin estimator for E[E[Y|X]^2] using K-fold CV.
    Returns a scalar computed out-of-fold.
    Supports 'rf', 'krr', and 'xgb' estimator types.
    """
    n_samples = X.shape[0]

    # Define model based on type
    if estimator_type == "rf":
        model_base = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    elif estimator_type == "krr":
        model_base = KernelRidge(kernel='rbf', alpha=0.1)
    elif estimator_type == "xgb":
        model_base = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=seed, n_jobs=-1, tree_method='hist') # Added XGBoost
    else:
        raise ValueError(f"Unsupported estimator_type: {estimator_type}. Choose 'rf', 'krr', or 'xgb'.")

    if n_folds <= 1:
        model = clone(model_base)
        model.fit(X, Y)
        mu_X = model.predict(X)
        return np.mean(mu_X ** 2)
    else:
        mu_X_all = np.zeros(n_samples)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train = Y[train_idx]

            model = clone(model_base) # Use clone for fresh model per fold
            model.fit(X_train, Y_train)
            mu_X_all[test_idx] = model.predict(X_test)
        return np.mean(mu_X_all ** 2)


def IF_estimator_conditional_mean(X, Y, estimator_type="rf",
                                  n_folds=5,
                                  k_neighbors_factor=0.1, # k as a fraction of n_samples
                                  min_k_neighbors=10,     # Minimum k value
                                  bandwidth_factor=0.1,
                                  seed=42):  # Factor for bandwidth heuristic
    """
    Computes the Influence Function (IF) based estimator for the conditional mean E[Y|X].

    Uses K-Fold cross-validation to mitigate bias from using the same data for
    model fitting and residual calculation. Adjusts k dynamically based on fold size.
    Supports 'rf', 'krr', and 'xgb' estimator types.

    Args:
        X (np.ndarray): Input features (n_samples, n_features).
        Y (np.ndarray): Outcome variable (n_samples,).
        estimator_type (str, optional): Base model type ('rf', 'krr', 'xgb'). Defaults to "rf".
        n_folds (int, optional): Number of folds for cross-validation.
                                 Set to <= 1 to disable CV. Defaults to 5.
        k_neighbors_factor (float, optional): Factor to determine default k
                                              (k = n_samples * factor). Defaults to 0.1.
        min_k_neighbors (int, optional): Minimum value for k neighbors. Defaults to 10.
        bandwidth_factor (float, optional): Factor for bandwidth heuristic
                                           (bw = factor * sqrt(n_features)). Defaults to 0.1.


    Returns:
        np.ndarray: Out-of-fold predictions for E[Y|X] (n_samples,).
                    Returns plugin predictions if CV is disabled or fails.
    """
    n_samples, n_features = X.shape
    out_preds = np.zeros(n_samples)
    bandwidth = bandwidth_factor * np.sqrt(n_features)
    if bandwidth < EPS:
        print(f"Warning: Calculated bandwidth is very small ({bandwidth}). Setting to EPS.")
        bandwidth = EPS
    default_k = max(min_k_neighbors, int(n_samples * k_neighbors_factor))

    # Define model base outside the loop/if conditions
    if estimator_type == "rf":
        model_base = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    elif estimator_type == "krr":
        model_base = KernelRidge(kernel='rbf', alpha=0.1)
    elif estimator_type == "xgb":
        model_base = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=seed, n_jobs=-1, tree_method='hist') # Added XGBoost
    else:
        raise ValueError(f"Unsupported estimator_type: {estimator_type}. Choose 'rf', 'krr', or 'xgb'.")

    # --- Case 1: No Cross-Validation (n_folds <= 1) ---
    if n_folds <= 1:
        print("Warning: Running IF estimator without cross-validation (n_folds <= 1).")
        k_actual = min(default_k, n_samples - 1 if n_samples > 1 else 1)
        if k_actual < 1:
             print("Error: Cannot perform k-NN correction with k < 1.")
             try:
                 model_fallback = clone(model_base)
                 model_fallback.fit(X, Y)
                 return model_fallback.predict(X)
             except Exception as e_plugin:
                 print(f"Error during fallback plugin calculation: {e_plugin}")
                 return np.full(n_samples, np.nan)

        # Fit base model
        model = None # Initialize
        try:
            model = clone(model_base)
            model.fit(X, Y)
            mu_X = model.predict(X)
            residuals = Y - mu_X
        except Exception as e_fit:
            print(f"Error fitting base model (no CV): {e_fit}")
            return np.full(n_samples, np.nan)

        # Perform k-NN correction
        try:
            scale = 1.0 / bandwidth
            tree = BallTree(X * scale, leaf_size=40)
            dist, ind = tree.query(X * scale, k=k_actual)
            W = np.exp(-0.5 * (dist**2))
            W_sum = W.sum(axis=1, keepdims=True)
            W_sum = np.where(W_sum < EPS, EPS, W_sum)
            W /= W_sum
            corrections = np.sum(W * residuals[ind], axis=1)
            out_preds = mu_X + corrections
        except Exception as e_corr:
             print(f"Error during k-NN correction calculation (no CV): {e_corr}")
             out_preds = mu_X # Fallback to plugin
    # --- Case 2: K-Fold Cross-Validation ---
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_num = 0
        for train_idx, test_idx in kf.split(X):
            fold_num += 1
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train = Y[train_idx]
            n_train = X_train.shape[0]

            if n_train < max(2, min_k_neighbors):
                 print(f"Warning: Training fold {fold_num} size ({n_train}) too small. Skipping IF correction, using fallback.")
                 try:
                      model_fold = clone(model_base)
                      model_fold.fit(X_train, Y_train)
                      out_preds[test_idx] = model_fold.predict(X_test)
                 except Exception as e_fallback:
                      print(f"  Error during fallback plugin calculation for fold {fold_num}: {e_fallback}")
                      out_preds[test_idx] = np.nan
                 continue

            k_fold = max(1, min(default_k, n_train - 1))
            model = None
            try:
                model = clone(model_base) # Use clone for fresh model per fold
                model.fit(X_train, Y_train)
                mu_test = model.predict(X_test)
                mu_train = model.predict(X_train)
                residuals_train = Y_train - mu_train

                scale = 1.0 / bandwidth
                tree = BallTree(X_train * scale, leaf_size=40)
                dist, ind = tree.query(X_test * scale, k=k_fold)
                W = np.exp(-0.5 * (dist**2))
                W_sum = W.sum(axis=1, keepdims=True)
                W_sum = np.where(W_sum < EPS, EPS, W_sum)
                W /= W_sum
                corrections = np.sum(W * residuals_train[ind], axis=1)
                out_preds[test_idx] = mu_test + corrections

            except ValueError as ve:
                 if "k must be less than or equal to the number of training points" in str(ve) or "k exceeds number of points" in str(ve):
                     print(f"Error during k-NN query in fold {fold_num} (k={k_fold}, n_train={n_train}): {ve}")
                     if model is not None:
                         try: out_preds[test_idx] = model.predict(X_test)
                         except: out_preds[test_idx] = np.nan
                     else: out_preds[test_idx] = np.nan
                 else:
                     print(f"ValueError during fold {fold_num} processing: {ve}")
                     out_preds[test_idx] = np.nan
            except Exception as e:
                 print(f"Error during fold {fold_num} processing: {e}")
                 if model is not None:
                     try: out_preds[test_idx] = model.predict(X_test)
                     except: out_preds[test_idx] = np.nan
                 else:
                     out_preds[test_idx] = np.nan

    return out_preds

def IF_estimator_squared_conditional(X, Y, estimator_type="rf", n_folds=N_FOLDS, seed=42):
    """
    Compute the IF-based estimator for E[E[Y|X]^2] using K-fold CV.
    Supports 'rf', 'krr', and 'xgb' estimator types.
    """
    if isinstance(X, Tensor): X = X.detach().cpu().numpy()
    if isinstance(Y, Tensor): Y = Y.detach().cpu().numpy()

    # Define model based on type
    if estimator_type == "rf":
        model_base = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    elif estimator_type == "krr":
        model_base = KernelRidge(kernel='rbf', alpha=0.1)
    elif estimator_type == "xgb":
        model_base = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=seed, n_jobs=-1, tree_method='hist') # Added XGBoost
    else:
        raise ValueError(f"Unsupported estimator_type: {estimator_type}. Choose 'rf', 'krr', or 'xgb'.")

    if n_folds <= 1:
        model = clone(model_base)
        model.fit(X, Y)
        mu_X = model.predict(X)
        plugin_estimate = np.mean(mu_X ** 2)
        residuals = Y - mu_X
        correction_term = 2 * np.mean(residuals * mu_X)
        return plugin_estimate + correction_term
    else:
        plugin_terms = []
        correction_terms = []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train = Y[train_idx]

            model = clone(model_base) # Use clone for fresh model per fold
            model.fit(X_train, Y_train)
            mu_X_test = model.predict(X_test)
            plugin_terms.append(np.mean(mu_X_test ** 2))
            residuals_test = Y[test_idx] - mu_X_test
            correction_terms.append(2 * np.mean(residuals_test * mu_X_test))

        # Handle potential NaNs if some folds failed? For now, assume they succeed.
        plugin_estimate = np.mean(plugin_terms) if plugin_terms else np.nan
        correction_term = np.mean(correction_terms) if correction_terms else np.nan

        if np.isnan(plugin_estimate) or np.isnan(correction_term):
            return np.nan
        else:
            return plugin_estimate + correction_term

# =============================================================================
# Kernel reweighting function (unchanged)
# =============================================================================

from dro_feature_selection.kernel_estimators import (
    chunked_pairwise_distance,
    estimate_conditional_expectation_knn,
    estimate_conditional_kernel_oof,
    estimate_conditional_keops,
    estimate_conditional_keops_flexible,
    estimate_conditional_keops_flexible_optimized,
    estimate_E_Y_S_kernel_flexible,
    estimate_T2_kernel_IF_like_flexible,
    estimate_T2_mc_flexible,
)

def analytical_gradient_penalty_alpha(alpha_torch: torch.Tensor,
                                      penalty_type: Optional[str],
                                      penalty_lambda: float) -> torch.Tensor:
    """Calculates analytical gradient dP/d(alpha) using autograd."""
    if penalty_type is None or penalty_lambda == 0 or penalty_type.lower() == "none":
        return torch.zeros_like(alpha_torch)

    alpha_torch_param = alpha_torch.clone().detach().requires_grad_(True)
    penalty_val = compute_penalty(alpha_torch_param, penalty_type, penalty_lambda)

    if penalty_val.requires_grad:
        try:
            grad_P = torch.autograd.grad(penalty_val, alpha_torch_param, retain_graph=False)[0]
            return grad_P
        except RuntimeError as e:
            print(f"Warning: Autograd failed for penalty {penalty_type}. Error: {e}")
            return torch.full_like(alpha_torch, float('nan')) # Indicate failure
    else:
        return torch.zeros_like(alpha_torch)
    


def estimate_gradient_autograd_flexible(X_std_torch: torch.Tensor,
                                        E_Yx_std_torch: torch.Tensor,
                                        param_torch: torch.Tensor, # Alpha or Theta (requires_grad=True)
                                        param_type: str,
                                        n_mc_samples: int,
                                        k_kernel: int,
                                        penalty_type: Optional[str],
                                        penalty_lambda: float) -> torch.Tensor:
    """
    Estimates total gradient dL/d(param) using Autograd through MC estimate of T2.
    L = T1_est - T2_est + P
    dL/dparam = -dT2/dparam + dP/dparam
    """
    if not param_torch.requires_grad:
        param_torch.requires_grad_(True)

    # --- Estimate T2 using MC + Flexible Kernel ---
    # This term needs to be differentiable w.r.t. param_torch
    term2_est = estimate_T2_mc_flexible(
        X_std_torch, E_Yx_std_torch, param_torch, param_type, n_mc_samples, k_kernel
    )

    # --- Calculate Penalty ---
    # Compute penalty based on alpha, deriving alpha from theta if needed
    if param_type == 'alpha':
        alpha_for_penalty = param_torch
    elif param_type == 'theta':
        # Need alpha = exp(theta) for penalty calculation, maintain grad graph
        alpha_for_penalty = torch.exp(param_torch)
    else:
        raise ValueError("param_type must be 'alpha' or 'theta'")

    penalty_value = compute_penalty(alpha_for_penalty, penalty_type, penalty_lambda)

    # --- Total Objective (for Autograd, ignoring constant T1) ---
    # L = -T2 + P (we minimize this, gradient is -dT2/dparam + dP/dparam)
    # Or L = T2 + P (if minimizing this, gradient is dT2/dparam + dP/dparam)
    # Preserve the original objective convention: minimize T2 + penalty.
    objective_L = term2_est + penalty_value

    # --- Compute Gradient ---
    grad_L = torch.autograd.grad(objective_L, param_torch, retain_graph=False)[0]

    return grad_L

def estimate_gradient_reinforce_flexible(X_std_torch: torch.Tensor,
                                         E_Yx_std_torch: torch.Tensor,
                                         param_torch: torch.Tensor, # Alpha or Theta (NO grad needed initially)
                                         param_type: str,
                                         n_grad_samples: int,
                                         k_kernel: int,
                                         penalty_type: Optional[str],
                                         penalty_lambda: float,
                                         use_baseline: bool) -> torch.Tensor:
    """
    Estimates total gradient dL/d(param) using REINFORCE for T2 part.
    L = T2 + P => dL/dparam = dT2/dparam + dP/dparam
    """
    device = param_torch.device
    m = param_torch.shape[0]
    # Use detached version for internal calculations not needing grad from param
    param_detached = param_torch.detach().clone()

    # --- REINFORCE Gradient for Term 2 ---
    grad_term2_accum = torch.zeros_like(param_torch)

    if param_type == 'alpha':
        alpha_clamped_detached = param_detached.clamp(CLAMP_MIN_ALPHA, CLAMP_MAX_ALPHA)
        noise_var = alpha_clamped_detached # Variance for noise generation
    elif param_type == 'theta':
        # Calculate alpha from theta for noise generation, clamp alpha
        alpha_detached = torch.exp(param_detached)
        alpha_clamped_detached = alpha_detached.clamp(CLAMP_MIN_ALPHA, CLAMP_MAX_ALPHA)
        noise_var = alpha_clamped_detached
    else:
        raise ValueError("param_type must be 'alpha' or 'theta'")

    for _ in range(n_grad_samples):
        with torch.no_grad(): # Sample noise and estimate reward g(S)^2
            epsilon_k = torch.randn_like(X_std_torch)
            S_param_k = X_std_torch + epsilon_k * torch.sqrt(noise_var)
            # Estimate g(S) = E[Y_std|S] using kernel, pass detached param
            g_hat_S_k = estimate_conditional_keops_flexible(
                X_std_torch, S_param_k, E_Yx_std_torch, param_detached, param_type, k=k_kernel
            )
            g_hat_S_k_squared = g_hat_S_k.pow(2) # Reward

        baseline = g_hat_S_k_squared.mean() if use_baseline else 0.0

        # Calculate score function: grad_param log p(S|param)
        if param_type == 'alpha':
            # Score for alpha: (epsilon^2 - 1) / (2 * alpha)
            score_term = (epsilon_k.pow(2) - 1.0) / (2.0 * alpha_clamped_detached + EPS)
        elif param_type == 'theta':
            # Score for theta: [(S-X)^2 - alpha] / (2 * alpha) = [(epsilon*sqrt(alpha))^2 - alpha]/(2*alpha)
            # score_term = (epsilon_k.pow(2) * alpha_clamped_detached - alpha_clamped_detached) / (2.0 * alpha_clamped_detached + EPS) # Simplify?
            # Original derivation: ((S-X)^2 - exp(theta)) / (2 * exp(theta))
            score_term = (epsilon_k.pow(2) * noise_var - noise_var) / (2.0 * noise_var + EPS)
            # score_term = (epsilon_k.pow(2) - 1.0) / 2.0 # Simplified version if noise_var cancels if > 0

        # Gradient estimate for E[g(S)^2] for this sample
        term_to_average = (g_hat_S_k_squared - baseline).unsqueeze(1) * score_term
        grad_term2_accum += term_to_average.mean(dim=0) # Average over batch N

    # Final REINFORCE gradient estimate for T2
    grad_term2_reinforce = grad_term2_accum / n_grad_samples

    # --- Penalty Gradient (using Autograd and Chain Rule if needed) ---
    alpha_torch_param = None
    if param_type == 'alpha':
        alpha_torch_param = param_torch.clone().detach().requires_grad_(True)
        grad_penalty_torch = analytical_gradient_penalty_alpha(alpha_torch_param, penalty_type, penalty_lambda)
    elif param_type == 'theta':
        theta_torch_param = param_torch.clone().detach().requires_grad_(True)
        alpha_from_theta = torch.exp(theta_torch_param)
        # Get grad w.r.t alpha first
        grad_penalty_alpha = analytical_gradient_penalty_alpha(alpha_from_theta, penalty_type, penalty_lambda)
        # Apply chain rule: dP/dtheta = dP/dalpha * dalpha/dtheta = dP/dalpha * alpha
        grad_penalty_torch = grad_penalty_alpha * alpha_from_theta # Element-wise product
    else:
        grad_penalty_torch = torch.zeros_like(param_torch)

    if torch.isnan(grad_penalty_torch).any():
         print(f"Warning: NaN detected in penalty gradient calculation for {param_type}.")
         grad_penalty_torch.nan_to_num_(0.0) # Replace NaN with 0 for safety

    # --- Total Gradient: dL/dparam = dT2/dparam + dP/dparam ---
    total_gradient = grad_term2_reinforce + grad_penalty_torch

    return total_gradient


def test_estimator(seeds, alpha_lists, X, Y, save_path=None):
    """
    Compares different estimators for the objective E[E[Y|X]^2] - E[E[Y|S]^2].

    Estimators for Term 2 (E[E[Y|S]^2]):
    - Plugin: plugin_estimator_squared_conditional(S, Y)
    - IF: IF_estimator_squared_conditional(S, Y)
    - IF-Plugin: Kernel(Plugin E[Y|X]) -> mean square
    - IF-IF: Kernel(IF E[Y|X]) -> mean square

    Term 1 (E[E[Y|X]^2]) is estimated using IF.

    Args:
        seeds (list): List of random seeds.
        alpha_lists (list): List of alpha vectors (or scalars if uniform noise).
                            Each element corresponds to one setting of alphas.
        X (Tensor or ndarray): Features.
        Y (Tensor or ndarray): Outcomes.
        save_path (str, optional): Path to save the comparison plot. Defaults to None.
    """
    # Sort the list of alphas by the maximum alpha value in each list/scalar
    alpha_lists = sorted(alpha_lists, key=lambda x: np.max(x) if isinstance(x, (np.ndarray, list)) else x)
    n = X.shape[0]  # Sample size
    eps_err = 1e-9 # Epsilon for division by zero in error calculation

    print(f"Running with seeds: {seeds}")
    print(f"Number of alpha settings: {len(alpha_lists)}")
    print(f"Sample size: {n}")

    # Storage: indexed by alpha setting index
    stats = {
        i: {
            'term_2_wrt_if': {'if_plugin': [], 'if_if': []}, # Compare Kernel Term2 estimates to IF Term2
            'objective_wrt_if': {'if_plugin': [], 'if_if': []} # Compare Obj(Kernel T2) to Obj(IF T2)
            # Add comparisons to Plugin if desired
            # 'term_2_wrt_plugin': {'if': [], 'if_plugin': [], 'if_if': []},
            # 'objective_wrt_plugin': {'if': [], 'if_plugin': [], 'if_if': []},
        }
        for i in range(len(alpha_lists))
    }

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed_idx+1}/{len(seeds)} ({seed}) ---")
        np.random.seed(seed)
        torch.manual_seed(seed) # Also seed torch

        # Ensure data are NumPy arrays for sklearn estimators
        if isinstance(X, Tensor): X_np = X.detach().cpu().numpy()
        else: X_np = np.array(X) # Ensure it's numpy
        if isinstance(Y, Tensor): Y_np = Y.detach().cpu().numpy()
        else: Y_np = np.array(Y) # Ensure it's numpy

        # --- Estimate Term 1 (using IF, assumed more stable/accurate) ---
        # This is constant for all alphas within a seed run
        print("    Estimating Term 1 (IF)...")
        if1 = IF_estimator_squared_conditional(X_np, Y_np, "rf", n_folds=N_FOLDS)
        if np.isnan(if1):
             print("    Term 1 (IF) calculation failed. Skipping seed.")
             continue

        for i, alpha_setting in enumerate(alpha_lists):
            # Handle both scalar alpha and vector alpha
            if isinstance(alpha_setting, (np.ndarray, list)):
                alpha = np.array(alpha_setting)
                alpha_max_str = f"{np.max(alpha):.4f}"
                alpha_min_str = f"{np.min(alpha):.4f}"
            else: # Assume scalar
                alpha = np.full(X_np.shape[1], alpha_setting) # Create vector
                alpha_max_str = f"{alpha_setting:.4f}"
                alpha_min_str = alpha_max_str

            print(f"\tAlpha setting {i}: Max={alpha_max_str}, Min={alpha_min_str}")

            # --- Generate S (Noisy Features) ---
            alpha_safe = np.maximum(alpha, 1e-12) # Ensure positivity
            noise = np.random.multivariate_normal(
                    mean=np.zeros(X_np.shape[1]),
                    cov=np.diag(alpha_safe),
                    size=n)
            S_np = X_np + noise

            # --- Convert relevant data to Tensors for Kernel estimator ---
            X_t = torch.from_numpy(X_np).float()
            S_t = torch.from_numpy(S_np).float()
            alpha_t = torch.from_numpy(alpha).float()

            # --- Estimate Term 2 variants ---
            print("      Estimating Term 2 variants...")
            try:
                # T2 Plugin: plugin_estimator_squared_conditional(S, Y)
                p2 = plugin_estimator_squared_conditional(S_np, Y_np, "rf", n_folds=N_FOLDS)

                # T2 IF: IF_estimator_squared_conditional(S, Y)
                if2 = IF_estimator_squared_conditional(S_np, Y_np, "rf", n_folds=N_FOLDS)

                # T2 IF-IF: Kernel(IF E[Y|X]) -> mean square
                E_Y_X_if = IF_estimator_conditional_mean(X_np, Y_np, "rf", n_folds=N_FOLDS)
                E_Y_X_if_t = torch.from_numpy(E_Y_X_if).float().to(X_t.device) # Move to same device
                E_Y_S_if = estimate_conditional_keops(X_t.to(E_Y_X_if_t.device), S_t.to(E_Y_X_if_t.device), E_Y_X_if_t, alpha_t.to(E_Y_X_if_t.device)).cpu().numpy()
                if2k = np.mean(E_Y_S_if**2)

                # T2 IF-Plugin: Kernel(Plugin E[Y|X]) -> mean square
                E_Y_X_plugin = plugin_estimator_conditional_mean(X_np, Y_np, "rf", n_folds=N_FOLDS)
                E_Y_X_plugin_t = torch.from_numpy(E_Y_X_plugin).float().to(X_t.device) # Move to same device
                E_Y_S_plugin = estimate_conditional_keops(X_t.to(E_Y_X_plugin_t.device), S_t.to(E_Y_X_plugin_t.device), E_Y_X_plugin_t, alpha_t.to(E_Y_X_plugin_t.device)).cpu().numpy()
                if2k_plugin = np.mean(E_Y_S_plugin**2)

            except Exception as e:
                 print(f"      Error during Term 2 estimation for alpha setting {i}: {e}")
                 # Store NaNs or skip this alpha setting for this seed
                 if i in stats: # Check if index exists
                     stats[i]['term_2_wrt_if']['if_plugin'].append(np.nan)
                     stats[i]['term_2_wrt_if']['if_if'].append(np.nan)
                     stats[i]['objective_wrt_if']['if_plugin'].append(np.nan)
                     stats[i]['objective_wrt_if']['if_if'].append(np.nan)
                 continue # Skip to next alpha

            # --- Calculate Objectives ---
            # Ensure Term 2 estimates are valid numbers before calculating objectives
            if np.isnan(if2) or np.isnan(if2k) or np.isnan(if2k_plugin):
                 print(f"      Skipping objective calculation due to NaN in Term 2 estimates.")
                 if i in stats:
                     stats[i]['term_2_wrt_if']['if_plugin'].append(np.nan if np.isnan(if2k_plugin) else abs(if2k_plugin - if2) / (abs(if2) + eps_err) * 100)
                     stats[i]['term_2_wrt_if']['if_if'].append(np.nan if np.isnan(if2k) else abs(if2k - if2) / (abs(if2) + eps_err) * 100)
                     stats[i]['objective_wrt_if']['if_plugin'].append(np.nan)
                     stats[i]['objective_wrt_if']['if_if'].append(np.nan)
                 continue

            ifobj = if1 - if2
            ifobjk = if1 - if2k
            ifobjk_plugin = if1 - if2k_plugin

            # --- Store Percentage Errors ---
            if2_denom = abs(if2) + eps_err
            ifobj_denom = abs(ifobj) + eps_err

            stats[i]['term_2_wrt_if']['if_plugin'].append(abs(if2k_plugin - if2) / if2_denom * 100)
            stats[i]['term_2_wrt_if']['if_if'].append(abs(if2k - if2) / if2_denom * 100)
            stats[i]['objective_wrt_if']['if_plugin'].append(abs(ifobjk_plugin - ifobj) / ifobj_denom * 100)
            stats[i]['objective_wrt_if']['if_if'].append(abs(ifobjk - ifobj) / ifobj_denom * 100)

            print(f"\t\tIF Obj: {ifobj:.4f}, IF-Plugin Obj: {ifobjk_plugin:.4f}, IF-IF Obj: {ifobjk:.4f}")
            print(f"\t\tIF T2: {if2:.4f}, IF-Plugin T2: {if2k_plugin:.4f}, IF-IF T2: {if2k:.4f}")


    # --- Aggregation and Plotting ---
    print("\n--- Aggregating and Plotting Results ---")
    # Use simple integer indices 1..N on the X‑axis
    x_positions = np.arange(1, len(alpha_lists) + 1)

    def mean_std(metric, method):
        # Handle cases where an alpha_max might not have results if script interrupted
        # Also handle potential NaNs stored during runs
        data_for_alphas = [stats[a][metric][method] for a in range(len(alpha_lists)) if a in stats and method in stats[a][metric]]

        if not data_for_alphas:
             print(f"Warning: No data found for metric='{metric}', method='{method}'. Skipping.")
             return np.full(len(alpha_lists), np.nan), np.full(len(alpha_lists), np.nan)

        # Calculate mean/std ignoring NaNs and handling potentially ragged lists if seeds failed
        means = []
        stds = []
        for alpha_data in data_for_alphas:
            valid_data = [d for d in alpha_data if not np.isnan(d)]
            if not valid_data:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(np.mean(valid_data))
                stds.append(np.std(valid_data))

        # Pad with NaNs if some alpha settings were skipped entirely
        if len(means) < len(alpha_lists):
             padded_means = np.full(len(alpha_lists), np.nan)
             padded_stds = np.full(len(alpha_lists), np.nan)
             # This assumes data_for_alphas corresponds to the first len(means) indices
             padded_means[:len(means)] = means
             padded_stds[:len(stds)] = stds
             return padded_means, padded_stds
        else:
             return np.array(means), np.array(stds)


    # Plotting setup
    metrics = ['term_2_wrt_if', 'objective_wrt_if'] # Focus on comparisons wrt IF
    method_map = {
        'term_2_wrt_if': ['if_plugin', 'if_if'],
        'objective_wrt_if': ['if_plugin', 'if_if']
    }
    styles = {
        'if_plugin':  {'fmt': '^--', 'label': 'Kernel(Plugin E[Y|X])'},
        'if_if':      {'fmt': 'x--', 'label': 'Kernel(IF E[Y|X])'}
    }

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), sharex=True)
    if len(metrics) == 1: axes = [axes] # Handle single metric case

    for ax, metric in zip(axes, metrics):
        for method in method_map[metric]:
            m, s = mean_std(metric, method)
            if np.isnan(m).all(): continue # Skip plotting if no data

            st = styles[method]
            label = f"{st['label']} (vs IF)" # Clarify baseline is IF

            ax.errorbar(x_positions, m, yerr=s, fmt=st['fmt'], capsize=5, label=label, alpha=0.8)

        title_parts = metric.replace('_wrt_if', ' Error wrt IF').replace('_', ' ').title()
        ax.set_title(title_parts)
        ax.set_ylabel('Percentage Error (%)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_yscale('log')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(i) for i in x_positions])

    axes[-1].set_xlabel('Index of Alpha Setting')
    fig.suptitle('Comparison of Kernel-based Estimators vs IF Estimator', fontsize=16)

    # Create subtitle with alpha setting details (showing max value)
    alpha_max_values = [f"{np.max(a):.2f}" if isinstance(a, (np.ndarray, list)) else f"{a:.2f}" for a in alpha_lists]
    subtitle = 'Alpha Setting Index -> Max Alpha Value:\n' + ', '.join([f"{i+1}:{val}" for i, val in enumerate(alpha_max_values)])
    fig.text(0.5, 0.01, subtitle, ha='center', va='bottom', fontsize=8) # Smaller font for subtitle

    fig.tight_layout(rect=[0, 0.05, 1, 0.96]) # Adjust rect for subtitle
    plot_filename = save_path or 'kernel_vs_if_comparison.png'
    fig.savefig(plot_filename, dpi=300)
    print(f"\nPlot saved to {plot_filename}")
    plt.close(fig)
