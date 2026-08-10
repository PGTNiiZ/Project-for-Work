import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import exp_r2_bert_feature as r2


class R2TrainingIntegrationTests(unittest.TestCase):
    def test_fresh_training_is_validation_only_and_writes_run_scoped_artifacts(self):
        rng = np.random.default_rng(17)
        n_train, n_calibration, n_ga, n_test = 40, 20, 20, 20
        n_total = n_train + n_calibration + n_ga + n_test
        pair_a = np.arange(n_total, dtype=np.int64)
        pair_b = pair_a + 1000
        actual = np.array(([0, 1] * (n_total // 2)), dtype=np.int8)
        cache = pd.DataFrame({
            "profile_id_a": pair_a,
            "profile_id_b": pair_b,
            "actual": actual,
            "split": ["val"] * (n_train + n_calibration + n_ga) + ["test"] * n_test,
            "experiment_split": (["model_train"] * n_train + ["model_calibration"] * n_calibration
                                 + ["ga_validation"] * n_ga + ["test"] * n_test),
            "decision_source": ["SCORED"] * n_total,
        })
        features = pd.DataFrame({"profile_id_a": pair_a, "profile_id_b": pair_b})
        for column in r2.FEATURE17_COLS:
            features[column] = rng.random(n_total, dtype=np.float32)
        bert_cos = pd.DataFrame({
            "profile_id_a": pair_a,
            "profile_id_b": pair_b,
            "bert_cos": rng.random(n_total, dtype=np.float32),
        })
        config = r2.R2TrainingConfig(seed=9, epochs=1, patience=1, batch_size=32,
                                     eval_batch_size=64, train_neg_ratio=1.0)
        with tempfile.TemporaryDirectory() as temp:
            output, metadata = r2.train_r2_probabilities(
                cache, features, bert_cos, Path(temp), training_config=config)
            root = Path(temp)
            self.assertEqual(len(output), n_total)
            self.assertEqual(metadata["split_counts"]["model_train"], n_train)
            self.assertEqual(metadata["split_counts"]["model_calibration"], n_calibration)
            self.assertEqual(metadata["n_predicted_all_scored_pairs"], n_total)
            for path in (root / "r2_probabilities.parquet", root / "r2_training.json",
                         root / "model" / "identity_mlp_r2.pt", root / "model" / "scaler.pkl",
                         root / "model" / "isotonic_calibrator.pkl", root / "model" / "feature_cols.pkl",
                         root / "training_history.csv"):
                self.assertTrue(path.is_file(), path)


if __name__ == "__main__":
    unittest.main()
