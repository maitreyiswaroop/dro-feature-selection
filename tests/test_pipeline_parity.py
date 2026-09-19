import tempfile
import unittest

import numpy as np
import torch

from dro_feature_selection.variable_selector import VariableSelector


class PaperPipelineCharacterizationTests(unittest.TestCase):
    @staticmethod
    def populations():
        X = torch.tensor(
            [
                [-1.0, 0.0],
                [-0.5, 0.5],
                [0.0, 1.0],
                [0.5, 0.5],
                [1.0, 0.0],
                [1.5, -0.5],
                [2.0, -1.0],
                [2.5, -0.5],
            ],
            dtype=torch.float32,
        )
        populations = []
        for pop_id, offset in enumerate((0.0, 0.2)):
            conditional_mean = 0.8 * X[:, 0] - 0.3 * X[:, 1] + offset
            outcome = conditional_mean + torch.linspace(-0.1, 0.1, len(X))
            populations.append(
                {
                    "pop_id": pop_id,
                    "X_std": X.clone(),
                    "Y_std": outcome,
                    "E_Yx_std": conditional_mean,
                    "term1_std": conditional_mean.pow(2).mean().item(),
                    "meaningful_indices": [0, 1],
                    "X_raw": X.numpy().copy(),
                    "Y_raw": outcome.numpy().copy(),
                }
            )
        return populations

    def test_selector_matches_paper_reference_output(self):
        common = {
            "parameterization": "alpha",
            "alpha_init": "random_1",
            "num_epochs": 2,
            "budget": 1,
            "penalty_type": "Reciprocal_L1",
            "penalty_lambda": 0.001,
            "learning_rate": 0.01,
            "optimizer_type": "adam",
            "early_stopping_patience": 15,
            "param_freezing": False,
            "smooth_minmax": float("inf"),
            "gradient_mode": "autograd",
            "t2_estimator_type": "kernel_if_like",
            "N_grad_samples": 1,
            "use_baseline": True,
            "estimator_type": "plugin",
            "base_model_type": "rf",
            "objective_value_estimator": "mc",
            "k_kernel": 3,
            "seed": 7,
        }

        with tempfile.TemporaryDirectory() as output_directory:
            params = dict(common)
            params["save_path"] = output_directory
            result = VariableSelector().run_selection(
                pop_data=self.populations(),
                m1=2,
                m=2,
                budget=common["budget"],
                params=params,
            )

        self.assertEqual(result["selected_indices"], [0])
        np.testing.assert_allclose(
            result["final_alpha"],
            [1.004679560661316, 1.06861412525177],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result["objective_history"],
            [0.38759320485405624, 0.48655313544441015],
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
