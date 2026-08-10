import unittest

import numpy as np

import exp_r1_ga_redecision as ga
from exp_lib import MATCH, NO_MATCH, REVIEW


class GATresholdUnitTests(unittest.TestCase):
    def test_decide_code_all_boundaries(self):
        genome = np.array([0.90, 0.70, 0.80, 0.40])
        prob = np.array([0.6999, 0.70, 0.70, 0.70, 0.70, 0.90])
        nsim = np.array([0.99, 0.80, 0.3999, 0.40, 0.7999, 0.00])
        actual = ga.decide_code(prob, nsim, genome)
        np.testing.assert_array_equal(actual, np.array([
            NO_MATCH, MATCH, NO_MATCH, REVIEW, REVIEW, MATCH,
        ], dtype=np.int8))

    def test_clip_preserves_bounds_and_ordering(self):
        genome = ga.clip(np.array([0.10, 2.0, 2.0, 3.0]))
        self.assertEqual(genome[0], 0.90)
        self.assertEqual(genome[1], 0.90)
        self.assertEqual(genome[2], 1.00)
        self.assertEqual(genome[3], 1.00)
        self.assertLessEqual(genome[1], genome[0])
        self.assertLessEqual(genome[3], genome[2])

    def test_cost_formula(self):
        decisions = np.array([MATCH, NO_MATCH, REVIEW, MATCH], dtype=np.int8)
        actual = np.array([0, 1, 0, 1], dtype=np.int8)
        weights = ga.CostWeights(5.0, 1.0, 0.02)
        self.assertEqual(ga.confusion_counts(decisions, actual), {"FP": 1, "FN": 1, "REVIEW": 1})
        self.assertAlmostEqual(ga.raw_cost(decisions, actual, weights), 6.02)

    def test_seed_reproduces_genome_and_history(self):
        prob = np.linspace(0.50, 0.995, 24)
        nsim = np.linspace(0.05, 0.99, 24)
        actual = np.array(([0, 1, 0, 0, 1, 0] * 4), dtype=np.int8)
        config = ga.GAConfig(seed=123, population_size=8, generations=5, elite_size=2,
                             mutation_probability=0.30, mutation_sigma=0.02)
        best_a, history_a = ga.run_ga(prob, nsim, actual, config=config, verbose=False)
        best_b, history_b = ga.run_ga(prob, nsim, actual, config=config, verbose=False)
        np.testing.assert_array_equal(best_a, best_b)
        self.assertEqual(history_a, history_b)

    def test_crossover_genes_come_from_a_or_b(self):
        parent_a = np.array([0.91, 0.72, 0.83, 0.14])
        parent_b = np.array([0.98, 0.88, 0.97, 0.43])
        child = ga.uniform_crossover(parent_a, parent_b, np.random.default_rng(7))
        for index, value in enumerate(child):
            self.assertIn(value, (parent_a[index], parent_b[index]))

    def test_mutation_changes_and_remains_valid(self):
        original = np.array([0.95, 0.80, 0.80, 0.20])
        config = ga.GAConfig(seed=1, population_size=8, generations=2, elite_size=2,
                             mutation_probability=1.0, mutation_sigma=0.10)
        mutated = ga.mutate(original, np.random.default_rng(7), config)
        self.assertFalse(np.array_equal(original, mutated))
        self.assertTrue(0.90 <= mutated[0] <= 0.999)
        self.assertTrue(0.50 <= mutated[1] <= mutated[0])
        self.assertTrue(0.50 <= mutated[2] <= 1.0)
        self.assertTrue(0.0 <= mutated[3] <= mutated[2])


if __name__ == "__main__":
    unittest.main()
