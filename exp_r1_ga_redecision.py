"""R1 GA re-decision with backward-compatible, configurable defaults.

Genome ``[t_m, t_r, c_promote, c_demote]`` controls the original 17-feature
score without training a new model.  The optimisation caller is responsible
for passing validation data only; this module has no knowledge of a test set.
"""
from dataclasses import dataclass, field

import numpy as np

from exp_lib import (EXP, MATCH, NO_MATCH, REVIEW, DEC_CODE, NSIM_MIN_SCORE,
                     build_cache, evaluate, save_json, split_constants)

# Keep these names for existing R2/R4/R6 scripts and the legacy command line.
W_FP, W_FN, W_REV = 5.0, 1.0, 0.02
POP, GENS, ELITE, SEED = 60, 40, 12, 42

STAGE18_MANUAL = np.array([0.98, 0.95, 0.90, 0.70])


@dataclass(frozen=True)
class CostWeights:
    """Business cost assigned to a false positive, false negative, and review."""

    fp: float = 5.0
    fn: float = 1.0
    review: float = 0.02

    def as_dict(self):
        return {"W_FP": self.fp, "W_FN": self.fn, "W_REV": self.review}


@dataclass(frozen=True)
class GAConfig:
    """Configuration for threshold optimisation; defaults reproduce legacy R1."""

    seed: int = 42
    population_size: int = 60
    generations: int = 40
    elite_size: int = 12
    mutation_probability: float = 0.30
    mutation_sigma: float = 0.02
    weights: CostWeights = field(default_factory=CostWeights)

    def validate(self):
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not 1 <= self.elite_size < self.population_size:
            raise ValueError("elite_size must be between 1 and population_size - 1")
        if self.generations < 1:
            raise ValueError("generations must be at least 1")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be between 0 and 1")
        if self.mutation_sigma < 0.0:
            raise ValueError("mutation_sigma must be non-negative")
        if min(self.weights.fp, self.weights.fn, self.weights.review) < 0.0:
            raise ValueError("cost weights must be non-negative")
        return self


def default_weights():
    """Read legacy globals so their historical edit-and-rerun behaviour remains."""
    return CostWeights(W_FP, W_FN, W_REV)


def default_config():
    return GAConfig(seed=SEED, population_size=POP, generations=GENS,
                    elite_size=ELITE, weights=default_weights())


def _config_or_default(config):
    return (default_config() if config is None else config).validate()


def decide_code(prob, nsim, g):
    """Return NO_MATCH/REVIEW/MATCH codes using the four-gene decision rule."""
    t_m, t_r, cp, cd = g
    d = np.full(len(prob), REVIEW, dtype=np.int8)
    d[prob < t_r] = NO_MATCH
    band = (prob >= t_r) & (prob < t_m)
    d[band & (nsim >= cp)] = MATCH
    d[band & (nsim < cd)] = NO_MATCH
    d[prob >= t_m] = MATCH
    return d


def confusion_counts(d, actual):
    """Count decisions on scored pairs; fixed split constants belong to exp_lib."""
    return {
        "FP": int(((d == MATCH) & (actual == 0)).sum()),
        "FN": int(((d == NO_MATCH) & (actual == 1)).sum()),
        "REVIEW": int((d == REVIEW).sum()),
    }


def cost_from_counts(counts, weights=None):
    """Compute ``W_FP*FP + W_FN*FN + W_REV*REVIEW``."""
    weights = default_weights() if weights is None else weights
    return (weights.fp * counts["FP"] + weights.fn * counts["FN"]
            + weights.review * counts["REVIEW"])


def raw_cost(d, actual, weights=None):
    return cost_from_counts(confusion_counts(d, actual), weights)


def clip(g):
    """Clip the genome to R1/R3 bounds while preserving both ordering constraints."""
    t_m = min(max(g[0], 0.90), 0.999)
    t_r = min(max(g[1], NSIM_MIN_SCORE), t_m)
    cp = min(max(g[2], 0.50), 1.0)
    cd = min(max(g[3], 0.0), cp)
    return np.array([t_m, t_r, cp, cd])


def uniform_crossover(parent_a, parent_b, rng):
    """Uniform crossover: each gene is copied from exactly one parent."""
    return np.where(rng.random(4) < 0.5, parent_a, parent_b)


def mutate(genome, rng, config=None):
    """Apply legacy-order Gaussian mutation, then clip to a valid genome."""
    cfg = _config_or_default(config)
    # The draw order is intentionally normal then mask: it preserves seed-42.
    delta = rng.normal(0, cfg.mutation_sigma, 4)
    mask = rng.random(4) < cfg.mutation_probability
    return clip(genome + delta * mask)


def run_ga(prob, nsim, actual, const_fn_below=0, config=None, verbose=True):
    """Optimise on the arrays supplied by the caller (validation data only)."""
    cfg = _config_or_default(config)
    rng = np.random.default_rng(cfg.seed)

    def cost(g):
        return (raw_cost(decide_code(prob, nsim, g), actual, cfg.weights)
                + cfg.weights.fn * const_fn_below)

    pop = [clip(rng.uniform([0.90, 0.50, 0.50, 0.0], [0.999, 0.90, 1.0, 0.6]))
           for _ in range(cfg.population_size)]
    history = []
    for gen in range(cfg.generations):
        pop.sort(key=cost)
        elite = pop[:cfg.elite_size]
        children = []
        while len(children) < cfg.population_size - cfg.elite_size:
            a, b = elite[rng.integers(cfg.elite_size)], elite[rng.integers(cfg.elite_size)]
            child = uniform_crossover(a, b, rng)
            children.append(mutate(child, rng, cfg))
        pop = elite + children
        best_c = cost(pop[0])
        history.append(dict(gen=gen, cost=float(best_c),
                            genome=[round(float(x), 4) for x in pop[0]]))
        if verbose and (gen % 5 == 0 or gen == cfg.generations - 1):
            print(f"  gen {gen:2d}  val cost {best_c:12,.1f}  g={np.round(pop[0], 4)}")
    return min(pop, key=cost), history


def main():
    """Retain the original one-off R1 command and legacy output locations."""
    print("Loading cache ...")
    cache = build_cache()
    consts = split_constants(cache)
    scored = cache[cache.decision_source != "AUTO_EXACT"]
    assert consts["full"]["total_pos"] == 29247, consts["full"]["total_pos"]

    results = {"weights": default_weights().as_dict(), "rules": {}}
    val = scored[scored.split == "val"]
    hi = val[val.score >= NSIM_MIN_SCORE]
    const_fn_below = int(((val.score < NSIM_MIN_SCORE) & (val.actual == 1)).sum())
    best, history = run_ga(hi.score.values, hi.name_sim.values,
                           hi.actual.values.astype(np.int8), const_fn_below)
    results["best_genome"] = dict(t_m=float(best[0]), t_r=float(best[1]),
                                  c_promote=float(best[2]), c_demote=float(best[3]))

    for sp in ["test", "full"]:
        sub = scored if sp == "full" else scored[scored.split == sp]
        prob, nsim = sub.score.values, sub.name_sim.values
        actual = sub.actual.values.astype(np.int8)
        prod_code = sub.decision.map(DEC_CODE).values.astype(np.int8)
        rules = {
            "R0_production": prod_code,
            "manual_stage18": decide_code(prob, nsim, STAGE18_MANUAL),
            "R1_ga": decide_code(prob, nsim, best),
        }
        results["rules"][sp] = {}
        for name, code in rules.items():
            metric = evaluate(code, actual, consts[sp])
            metric["cost"] = round(float(cost_from_counts(metric)), 1)
            results["rules"][sp][name] = metric

    save_json(EXP / "r1_results.json", results)
    save_json(EXP / "r1_ga_history.json", history)
    print(f"Saved -> {EXP / 'r1_results.json'}, {EXP / 'r1_ga_history.json'}")


if __name__ == "__main__":
    main()
