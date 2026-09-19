import unittest

import torch

from estimators import (
    estimate_conditional_keops_flexible,
    estimate_conditional_keops_flexible_optimized,
)


class KernelEstimatorCharacterizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.X = torch.tensor(
            [
                [-1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.5],
                [2.0, -0.5],
            ],
            dtype=torch.float64,
        )
        self.S = torch.tensor(
            [[-0.5, 0.25], [0.75, 0.75], [1.75, -0.25]],
            dtype=torch.float64,
        )
        self.conditional_means = torch.tensor(
            [-1.5, 0.25, 1.0, 2.5], dtype=torch.float64
        )

    @staticmethod
    def reference_estimate(X, S, conditional_means, alpha, k):
        scale = torch.rsqrt(alpha.clamp(min=1e-4))
        distances = torch.cdist(S * scale, X * scale).pow(2)
        nearest_distances, nearest_indices = torch.topk(
            distances, min(k, X.shape[0]), dim=1, largest=False
        )
        weights = torch.exp(-0.5 * nearest_distances)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return (weights * conditional_means[nearest_indices]).sum(dim=1)

    def test_alpha_parameterization_matches_reference(self):
        alpha = torch.tensor([0.5, 2.0], dtype=torch.float64)
        expected = self.reference_estimate(
            self.X, self.S, self.conditional_means, alpha, k=3
        )

        actual = estimate_conditional_keops_flexible(
            self.X,
            self.S,
            self.conditional_means,
            alpha,
            param_type="alpha",
            k=3,
            chunk_size=1,
        )

        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-7)

    def test_alpha_and_theta_parameterizations_are_equivalent(self):
        alpha = torch.tensor([0.5, 2.0], dtype=torch.float64)
        theta = torch.log(alpha)

        from_alpha = estimate_conditional_keops_flexible(
            self.X, self.S, self.conditional_means, alpha, "alpha", k=3
        )
        from_theta = estimate_conditional_keops_flexible(
            self.X, self.S, self.conditional_means, theta, "theta", k=3
        )

        torch.testing.assert_close(from_alpha, from_theta, rtol=1e-7, atol=1e-7)

    def test_optimized_and_chunked_implementations_agree(self):
        alpha = torch.tensor([0.5, 2.0], dtype=torch.float64)

        chunked = estimate_conditional_keops_flexible(
            self.X, self.S, self.conditional_means, alpha, "alpha", k=3
        )
        optimized = estimate_conditional_keops_flexible_optimized(
            self.X,
            self.S,
            self.conditional_means,
            alpha,
            param_type="alpha",
            k=3,
            max_batch_size=1,
        )

        torch.testing.assert_close(chunked, optimized, rtol=1e-7, atol=1e-7)

    def test_kernel_output_retains_parameter_gradient(self):
        alpha = torch.tensor([0.5, 2.0], dtype=torch.float64, requires_grad=True)
        estimate = estimate_conditional_keops_flexible(
            self.X, self.S, self.conditional_means, alpha, "alpha", k=3
        )

        estimate.sum().backward()

        self.assertIsNotNone(alpha.grad)
        self.assertTrue(torch.isfinite(alpha.grad).all())
        self.assertGreater(torch.linalg.vector_norm(alpha.grad).item(), 0.0)

    def test_k_is_capped_at_number_of_reference_points(self):
        alpha = torch.tensor([1.0, 1.0], dtype=torch.float64)

        capped = estimate_conditional_keops_flexible(
            self.X, self.S, self.conditional_means, alpha, "alpha", k=len(self.X)
        )
        oversized = estimate_conditional_keops_flexible(
            self.X, self.S, self.conditional_means, alpha, "alpha", k=100
        )

        torch.testing.assert_close(capped, oversized, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
