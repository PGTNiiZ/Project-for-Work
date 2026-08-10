import hashlib
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import exp_r1_ga_redecision as ga
import exp_r2_bert_feature as r2
import run_ga_experiments as runner
from exp_lib import build_nested_entity_split


KEYS = ["profile_id_a", "profile_id_b"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nested_training_inputs(seed: int = 11):
    """Small fully-labelled cache whose split assignment is explicit for trainer tests."""
    rng = np.random.default_rng(seed)
    sizes = {"model_train": 40, "model_calibration": 20, "ga_validation": 20, "test": 20}
    roles = sum(([role] * count for role, count in sizes.items()), [])
    total = len(roles)
    a = np.arange(total, dtype=np.int64)
    b = a + 10_000
    actual = np.array([0, 1] * (total // 2), dtype=np.int8)
    cache = pd.DataFrame({
        "profile_id_a": a,
        "profile_id_b": b,
        "actual": actual,
        "split": ["test" if role == "test" else "val" for role in roles],
        "experiment_split": roles,
        "decision_source": "SCORED",
    })
    feature = pd.DataFrame({"profile_id_a": a, "profile_id_b": b})
    for col in r2.FEATURE17_COLS:
        # A deliberately distinct distribution makes scaler-fit leakage detectable.
        feature[col] = np.where(np.asarray(roles) == "model_train", 0.1, 0.9).astype(np.float32)
    feature["username_jaro"] = np.where(actual == 1, 0.2, 0.0).astype(np.float32)
    bert = pd.DataFrame({"profile_id_a": a, "profile_id_b": b,
                         "bert_cos": rng.uniform(-0.2, 0.8, total).astype(np.float32)})
    return cache, feature, bert


class NestedEntitySplitTests(unittest.TestCase):
    def test_nested_split_is_deterministic_entity_disjoint_and_drops_cross_role_pairs(self):
        # Every entity owns two profiles, so the input itself contains no accidental entity leakage.
        n_entities = 300
        ids_a = np.arange(n_entities, dtype=np.int64) * 2
        ids_b = ids_a + 1
        keys = pd.Series(
            [f"entity-{index}" for index in range(n_entities) for _ in range(2)],
            index=np.arange(n_entities * 2, dtype=np.int64),
        )
        cache = pd.DataFrame({
            "profile_id_a": ids_a,
            "profile_id_b": ids_b,
            "split": ["val"] * 260 + ["test"] * 40,
            "decision_source": "SCORED",
            "actual": [0, 1] * 150,
        })
        first, manifest = build_nested_entity_split(cache, seed=73, entity_keys=keys)
        second, _ = build_nested_entity_split(cache, seed=73, entity_keys=keys)
        pd.testing.assert_series_equal(first["experiment_split"], second["experiment_split"])
        self.assertEqual(manifest["entity_overlap_check"], "passed")
        self.assertGreater(manifest["pair_counts"]["model_train"], 0)
        self.assertGreater(manifest["pair_counts"]["model_calibration"], 0)
        self.assertGreater(manifest["pair_counts"]["ga_validation"], 0)

        entity_sets = {}
        for role in ("model_train", "model_calibration", "ga_validation", "test"):
            subset = first.loc[first["experiment_split"] == role]
            entity_sets[role] = set(keys.loc[subset["profile_id_a"]]) | set(keys.loc[subset["profile_id_b"]])
        roles = list(entity_sets)
        for index, left in enumerate(roles):
            for right in roles[index + 1:]:
                self.assertFalse(entity_sets[left] & entity_sets[right])

        train_entity = next(iter(entity_sets["model_train"]))
        ga_entity = next(iter(entity_sets["ga_validation"]))
        train_pid = int(keys[keys == train_entity].index[0])
        ga_pid = int(keys[keys == ga_entity].index[0])
        cross = pd.concat([cache, pd.DataFrame([{
            "profile_id_a": train_pid, "profile_id_b": ga_pid, "split": "val",
            "decision_source": "SCORED", "actual": 0,
        }])], ignore_index=True)
        with_cross, _ = build_nested_entity_split(cross, seed=73, entity_keys=keys)
        self.assertEqual(with_cross.iloc[-1]["experiment_split"], "drop")


class R2AntiLeakageTests(unittest.TestCase):
    def test_scaler_is_fit_from_model_train_only(self):
        cache, feature, bert = nested_training_inputs()
        cfg = r2.R2TrainingConfig(seed=19, epochs=1, patience=1, batch_size=32,
                                  eval_batch_size=64, train_neg_ratio=1.0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            r2.train_r2_probabilities(cache, feature, bert, root, cfg)
            with (root / "model" / "scaler.pkl").open("rb") as handle:
                scaler = pickle.load(handle)
        expected = feature.merge(bert, on=KEYS, validate="one_to_one").loc[
            cache["experiment_split"] == "model_train", r2.FEATURE17_COLS + ["bert_cos"]]
        np.testing.assert_allclose(scaler.mean_, expected.mean(axis=0).to_numpy(), atol=1e-7)

    def test_ga_validation_and_test_labels_do_not_change_model_or_probabilities(self):
        cache, feature, bert = nested_training_inputs()
        changed = cache.copy()
        changed.loc[changed["experiment_split"].isin(["ga_validation", "test"]), "actual"] ^= 1
        cfg = r2.R2TrainingConfig(seed=29, epochs=1, patience=1, batch_size=32,
                                  eval_batch_size=64, train_neg_ratio=1.0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, _ = r2.train_r2_probabilities(cache, feature, bert, root / "first", cfg)
            second, _ = r2.train_r2_probabilities(changed, feature, bert, root / "second", cfg)
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(sha256(root / "first" / "model" / "identity_mlp_r2.pt"),
                             sha256(root / "second" / "model" / "identity_mlp_r2.pt"))
            first_history = pd.read_csv(root / "first" / "training_history.csv").drop(columns=["elapsed_seconds"])
            second_history = pd.read_csv(root / "second" / "training_history.csv").drop(columns=["elapsed_seconds"])
            pd.testing.assert_frame_equal(first_history, second_history)

    def test_duplicate_feature_keys_are_rejected_before_training(self):
        cache, feature, bert = nested_training_inputs()
        duplicate = pd.concat([feature, feature.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "duplicate pair keys"):
                r2.train_r2_probabilities(cache, duplicate, bert, Path(temp),
                                           r2.R2TrainingConfig(epochs=1, patience=1, batch_size=32))


class R3AntiLeakageTests(unittest.TestCase):
    def test_test_labels_do_not_change_ga_genome(self):
        validation = pd.DataFrame({
            "profile_id_a": range(8), "profile_id_b": range(100, 108),
            "experiment_split": ["ga_validation"] * 4 + ["test"] * 4,
            "name_sim": [0.1, 0.9, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0],
            "actual": [0, 1, 0, 1, 0, 0, 0, 0],
            "decision_source": "SCORED",
        })
        probabilities = pd.DataFrame({
            "profile_id_a": range(8), "profile_id_b": range(100, 108),
            "probability_r2": [0.50, 0.95, 0.60, 0.99, 0.1, 0.1, 0.1, 0.1],
        })
        constants = {
            "ga_validation": {"tp_exact": 0, "fp_exact": 0, "fn_blocking": 0, "total_pos": 2, "n_scored": 4},
            "test": {"tp_exact": 0, "fp_exact": 0, "fn_blocking": 0, "total_pos": 4, "n_scored": 4},
        }
        settings = {"population_size": 8, "generations": 3, "elite_size": 2,
                    "mutation_probability": 0.3, "mutation_sigma": 0.02}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prob_path = root / "r2_probabilities.parquet"
            probabilities.to_parquet(prob_path, index=False)
            one = runner.make_r3_data(validation, prob_path)
            changed = validation.copy()
            changed.loc[changed["experiment_split"] == "test", "actual"] = 1
            two = runner.make_r3_data(changed, prob_path)
            result_one = runner.run_experiments(one, constants, [41], settings, [("A", ga.CostWeights())], root / "one")
            result_two = runner.run_experiments(two, constants, [41], settings, [("A", ga.CostWeights())], root / "two")
        for key in ("t_m", "t_r", "c_promote", "c_demote", "val_cost", "objective_val_cost"):
            self.assertEqual(result_one["trials"][0][key], result_two["trials"][0][key])


if __name__ == "__main__":
    unittest.main()
