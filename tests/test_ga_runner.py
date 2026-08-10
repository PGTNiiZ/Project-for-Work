import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import exp_r1_ga_redecision as ga
import run_ga_experiments as runner


def synthetic_r1_data(test_actual):
    """A small data set whose test labels can change without changing validation."""
    val_scores = [0.50, 0.55, 0.72, 0.78, 0.85, 0.91, 0.96, 0.995]
    val_nsim = [0.10, 0.95, 0.15, 0.85, 0.30, 0.90, 0.55, 0.20]
    val_actual = [0, 1, 0, 1, 0, 1, 0, 1]
    test_scores = [0.10] * len(test_actual)  # always NO_MATCH; never affects GA search.
    frame = pd.DataFrame({
        "split": ["val"] * len(val_scores) + ["test"] * len(test_actual),
        "score": val_scores + test_scores,
        "name_sim": val_nsim + [0.0] * len(test_actual),
        "actual": val_actual + list(test_actual),
        "decision": ["NO_MATCH"] * (len(val_scores) + len(test_actual)),
        "decision_source": ["SCORED"] * (len(val_scores) + len(test_actual)),
    })
    constants = {
        "val": {"tp_exact": 0, "fp_exact": 0, "fn_blocking": 0, "total_pos": sum(val_actual), "n_scored": len(val_scores)},
        "test": {"tp_exact": 0, "fp_exact": 0, "fn_blocking": 0, "total_pos": sum(test_actual), "n_scored": len(test_actual)},
    }
    return runner.make_r1_data(frame), constants


class GARunnerIntegrationTests(unittest.TestCase):
    def test_r3_loads_existing_probabilities_without_training(self):
        cache = pd.DataFrame({
            "profile_id_a": [1, 2, 3, 4],
            "profile_id_b": [11, 12, 13, 14],
            "split": ["val", "val", "test", "test"],
            "experiment_split": ["ga_validation", "ga_validation", "test", "test"],
            "name_sim": [0.2, 0.9, 0.1, 0.8],
            "actual": [0, 1, 0, 1],
            "decision_source": ["SCORED"] * 4,
        })
        probabilities = pd.DataFrame({
            "profile_id_a": [1, 2, 3, 4],
            "profile_id_b": [11, 12, 13, 14],
            "probability_r2": [0.50, 0.99, 0.96, 0.10],
        })
        with tempfile.TemporaryDirectory() as temp:
            probability_path = Path(temp) / "r2_probabilities.parquet"
            probabilities.to_parquet(probability_path, index=False)
            data = runner.make_r3_data(cache, probability_path)
        self.assertEqual(data.experiment, "r3")
        self.assertEqual(data.baseline_label, "R2_manual")
        np.testing.assert_array_equal(data.baseline_decider(data.scored), np.array([0, 2, 1, 0], dtype=np.int8))

    def test_sensitivity_scenarios_are_fixed_a_b_c(self):
        scenarios = runner._scenario_specs("r1-sensitivity", None)
        self.assertEqual([name for name, _ in scenarios], ["A_current", "B_fp10", "C_fn2"])
        self.assertEqual([weights.as_dict() for _, weights in scenarios], [
            {"W_FP": 5.0, "W_FN": 1.0, "W_REV": 0.02},
            {"W_FP": 10.0, "W_FN": 1.0, "W_REV": 0.02},
            {"W_FP": 5.0, "W_FN": 2.0, "W_REV": 0.02},
        ])

    def test_validation_only_selection_and_complete_outputs(self):
        settings = {
            "population_size": 8,
            "generations": 3,
            "elite_size": 2,
            "mutation_probability": 0.30,
            "mutation_sigma": 0.02,
        }
        scenarios = [("A_current", ga.CostWeights())]
        data_a, consts_a = synthetic_r1_data([1] + [0] * 7)
        data_b, consts_b = synthetic_r1_data([1] * 8)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary_a = runner.run_experiments(data_a, consts_a, [19], settings, scenarios,
                                               root / "first", run_id="first")
            summary_b = runner.run_experiments(data_b, consts_b, [19], settings, scenarios,
                                               root / "second", run_id="second")
            first_trial = summary_a["trials"][0]
            second_trial = summary_b["trials"][0]

            # Test labels differ, but GA selection and history-driving validation cost cannot.
            for key in ("t_m", "t_r", "c_promote", "c_demote", "val_cost", "objective_val_cost"):
                self.assertEqual(first_trial[key], second_trial[key])
            self.assertNotEqual(first_trial["test_cost"], second_trial["test_cost"])

            output = root / "first"
            for name in ("config.json", "trials.csv", "generations.csv", "summary.json", "summary.md"):
                self.assertTrue((output / name).is_file(), name)
            with (output / "trials.csv").open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            for key in ("experiment", "seed", "t_m", "t_r", "c_promote", "c_demote", "val_cost",
                        "test_TP", "test_FP", "test_FN", "test_REVIEW", "test_precision", "test_recall",
                        "test_F1", "test_cost", "runtime_seconds"):
                self.assertIn(key, row)
            with (output / "config.json").open(encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["seeds"], [19])


@unittest.skipUnless(Path("experiments/scored_pairs_enriched.parquet").is_file(), "real R1 cache is unavailable")
class GARegressionTests(unittest.TestCase):
    def test_default_seed_42_reproduces_r1_artifact(self):
        cache = runner.build_cache()
        data = runner.make_r1_data(cache)
        consts = runner.split_constants(cache)
        val = data.scored.loc[data.scored["split"] == "val"]
        hi = val.loc[val["score"] >= runner.NSIM_MIN_SCORE]
        const_fn_below = int(((val["score"] < runner.NSIM_MIN_SCORE) & (val["actual"] == 1)).sum())
        best, history = ga.run_ga(hi["score"].to_numpy(), hi["name_sim"].to_numpy(),
                                  hi["actual"].to_numpy(dtype=np.int8), const_fn_below,
                                  verbose=False)
        np.testing.assert_allclose(best, np.array([0.9990, 0.9642, 0.9875, 0.5052]), atol=0.0002)
        self.assertEqual(len(history), 40)

        test = data.scored.loc[data.scored["split"] == "test"]
        metrics = runner._metric_with_cost(
            ga.decide_code(test["score"].to_numpy(), test["name_sim"].to_numpy(), best),
            test["actual"].to_numpy(), consts["test"], ga.default_weights(),
        )
        self.assertAlmostEqual(metrics["F1"], 0.8292, places=4)
        self.assertAlmostEqual(metrics["cost"], 2062.3, delta=0.1)


if __name__ == "__main__":
    unittest.main()
