import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dro_feature_selection.baselines import (
    baseline_dro_lasso_comparison,
    baseline_lasso_comparison,
)
from dro_feature_selection.data_loader import DataManager
from dro_feature_selection.downstream_eval import DownstreamEvaluator


class RealDataPipelineTests(unittest.TestCase):
    @staticmethod
    def classification_populations():
        rng = np.random.default_rng(7)
        populations = []
        for name in ("Male", "Female"):
            X = rng.normal(size=(20, 4))
            Y = (X[:, 0] > 0).astype(float)
            populations.append(
                {
                    "pop_id": name,
                    "X_raw": X,
                    "Y_raw": Y,
                    "meaningful_indices": None,
                }
            )
        return populations

    def test_uci_pipeline_preprocesses_multiple_populations(self):
        rng = np.random.default_rng(7)
        raw_populations = [
            {
                "pop_id": name,
                "X_raw": rng.normal(size=(20, 4)),
                "Y_raw": np.tile([0.0, 1.0], 10),
                "meaningful_indices": None,
            }
            for name in ("Male", "Female")
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = DataManager(
                cache_dir=f"{temporary_directory}/cache",
                raw_data_dir=f"{temporary_directory}/raw",
            )
            with patch(
                "dro_feature_selection.data.uci.get_uci_pop_data",
                return_value=raw_populations,
            ), patch(
                "dro_feature_selection.estimators.plugin_estimator_conditional_mean",
                side_effect=lambda X, Y, *args, **kwargs: Y.astype(float),
            ):
                train, test = manager._process_uci_data(
                    ["Male", "Female"],
                    seed=7,
                    estimator_type="plugin",
                    device="cpu",
                    base_model_type="rf",
                    is_classification=True,
                )

        self.assertEqual([population["pop_id"] for population in train], ["Male", "Female"])
        self.assertEqual([tuple(population["X_std"].shape) for population in train], [(12, 4), (12, 4)])
        self.assertEqual([tuple(population["X_std"].shape) for population in test], [(8, 4), (8, 4)])
        self.assertTrue(all(np.isfinite(population["term1_std"]) for population in train))

    def test_acs_pipeline_uses_portable_paths_and_preprocesses_state(self):
        rng = np.random.default_rng(7)
        feature_names = ["PERNP", "INTP", "RETP"]
        X = rng.normal(size=(20, len(feature_names)))
        Y = rng.normal(size=20)
        generated_data = (X, Y, [X.copy()], [Y.copy()], feature_names, ["CA"])

        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = DataManager(
                cache_dir=f"{temporary_directory}/cache",
                raw_data_dir=f"{temporary_directory}/raw",
            )
            with patch(
                "dro_feature_selection.data.acs.generate_data_acs",
                return_value=generated_data,
            ) as generate_data, patch(
                "dro_feature_selection.estimators.plugin_estimator_conditional_mean",
                side_effect=lambda X, Y, *args, **kwargs: Y.astype(float),
            ):
                train, test = manager._generate_acs_data(
                    pop_configs=[{"pop_id": 0, "dataset_type": "acs"}],
                    m1=2,
                    m=3,
                    dataset_size=20,
                    acs_data_fraction=1.0,
                    estimator_type="plugin",
                    device="cpu",
                    base_model_type="rf",
                    seed=7,
                    acs_states=["CA"],
                )

        self.assertEqual(generate_data.call_args.kwargs["states"], ["CA"])
        self.assertNotIn("/data/user_data/", generate_data.call_args.kwargs["root_dir"])
        self.assertEqual(train[0]["pop_id"], "CA")
        self.assertEqual(tuple(train[0]["X_std"].shape), (12, 3))
        self.assertEqual(tuple(test[0]["X_std"].shape), (8, 3))
        self.assertTrue(np.isfinite(train[0]["term1_std"]))

    def test_lasso_baselines_use_classifiers_for_uci_data(self):
        populations = self.classification_populations()
        pooled = baseline_lasso_comparison(
            populations,
            budget=2,
            alpha_lasso=0.01,
            classification=True,
            seed=7,
        )
        robust = baseline_dro_lasso_comparison(
            populations,
            budget=2,
            alpha_lasso=0.01,
            classification=True,
            max_iter=2,
            seed=7,
        )

        self.assertEqual(len(pooled["selected_indices"]), 2)
        self.assertEqual(len(robust["selected_indices"]), 2)

    def test_downstream_classification_handles_single_class_split(self):
        X = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        population = {
            "pop_id": "Female",
            "X_std": X,
            "Y_std": torch.zeros(6),
        }
        results = DownstreamEvaluator()._evaluate_method(
            [population],
            {"Female": (np.array([0, 1, 2, 3]), np.array([4, 5]))},
            selected_indices=[0, 1],
            method_name="test",
            seed=7,
            is_classification=True,
        )

        self.assertEqual(len(results), 1)
        self.assertTrue(np.isfinite(results[0]["logloss"]))


if __name__ == "__main__":
    unittest.main()
