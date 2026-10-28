# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Evolve Frog CPG gaits for language-specified behaviours (step 1).

Each prompt is compiled (by hand for now; an LLM compiler comes later) into a
metric over the behaviour descriptors returned by frog.rollout. A real-valued
GA (tournament 3, BLX-0.5 crossover, Gaussian mutation, 10% elitism) maximises
the metric. Flipping over costs a fixed penalty, so a gait must stay upright
for the whole episode to win.

After evolution every champion is scored on EVERY metric (specificity table):
a behaviour counts only if its champion is best on its own metric.

Usage: python evolve.py [seed] [generations]
Outputs: results/<behaviour>/{best_genome.json,history.json,summary.json},
         results/specificity.json, gifs/<behaviour>.gif
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import frog

HERE = os.path.dirname(os.path.abspath(__file__))
POP, ELITE, TOURN = 64, 6, 3
MUT_RATE, MUT_SIGMA = 0.2, 0.15
FLIP_PENALTY = 2.0
N_WORKERS = int(os.environ.get("N_WORKERS", "8"))

BEHAVIOURS = {
    "forward":   ("walk forward",                   lambda d: d["dx"]),
    "backward":  ("walk backward",                  lambda d: -d["dx"]),
    "sideways":  ("walk sideways to the left",      lambda d: d["dy"] - 0.5 * abs(d["dx"])),
    "turn_left": ("turn left on the spot",          lambda d: d["yaw"] - 2.0 * np.hypot(d["dx"], d["dy"])),
    "hop":       ("hop: keep all feet off the ground as much as possible", lambda d: d["airtime"]),
}


def fitness(descs, key):
    """Worst case over the friction conditions (robust to physics details)."""
    return min(BEHAVIOURS[key][1](d) - (FLIP_PENALTY if d["flipped_at"] is not None else 0.0) for d in descs)


def _eval(genome):
    return [frog.rollout(genome, friction=mu) for mu in frog.FRICTIONS]


def evolve(key, seed, generations, pool):
    rng = np.random.default_rng(seed)
    pop = rng.random((POP, frog.N_GENES))
    history, best = [], (-np.inf, None, None)
    t0 = time.time()
    for gen in range(generations):
        descs = list(pool.map(_eval, pop, chunksize=4))
        fit = np.array([fitness(d, key) for d in descs])
        descs = [d[0] for d in descs]                     # report the nominal (friction 1.0) episode
        order = np.argsort(fit)[::-1]
        pop, fit, descs = pop[order], fit[order], [descs[i] for i in order]
        if fit[0] > best[0]:
            best = (float(fit[0]), pop[0].copy(), descs[0])
        history.append({"generation": gen, "best": float(fit[0]), "mean": float(fit.mean()),
                        "best_desc": descs[0]})
        if gen % 10 == 0 or gen == generations - 1:
            print(f"  [{key}] gen {gen:3d} best {fit[0]:7.3f} mean {fit.mean():7.3f} "
                  f"{time.time() - t0:5.0f}s", flush=True)
        children = [pop[i].copy() for i in range(ELITE)]
        while len(children) < POP:
            a = pop[min(rng.integers(0, POP, TOURN))]
            b = pop[min(rng.integers(0, POP, TOURN))]
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            span = hi - lo
            c = rng.uniform(lo - 0.5 * span, hi + 0.5 * span)
            m = rng.random(frog.N_GENES) < MUT_RATE
            c[m] += rng.normal(0, MUT_SIGMA, m.sum())
            children.append(np.clip(c, 0, 1))
        pop = np.array(children)
    return best, history, time.time() - t0


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    generations = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    out = os.path.join(HERE, "results")
    champions = {}
    with ProcessPoolExecutor(N_WORKERS) as pool:
        for key, (prompt, _) in BEHAVIOURS.items():
            print(f"== {key}: \"{prompt}\"", flush=True)
            (fit, genome, desc), history, wall = evolve(key, seed, generations, pool)
            d = os.path.join(out, key)
            os.makedirs(d, exist_ok=True)
            json.dump(genome.tolist(), open(os.path.join(d, "best_genome.json"), "w"))
            json.dump(history, open(os.path.join(d, "history.json"), "w"))
            f, A, phi, b = frog.decode(genome)
            json.dump({"behaviour": key, "prompt": prompt, "seed": seed, "generations": generations,
                       "pop": POP, "fitness": fit, "descriptors": desc, "wall_clock_s": wall,
                       "evaluations": POP * generations, "sim_steps_per_eval": int(frog.SECONDS / frog.DT),
                       "cpg": {"f_hz": f, "A": A.tolist(), "phi": phi.tolist(), "b": b.tolist()}},
                      open(os.path.join(d, "summary.json"), "w"), indent=1)
            champions[key] = genome

    # specificity: every champion scored on every metric
    table = {}
    for key, g in champions.items():
        desc = frog.rollout(g)
        table[key] = {"descriptors": desc,
                      "scores": {k: float(BEHAVIOURS[k][1](desc)) for k in BEHAVIOURS}}
    json.dump(table, open(os.path.join(out, "specificity.json"), "w"), indent=1)
    print("\nspecificity (row = champion evolved for, column = metric)")
    print(" " * 11 + "".join(f"{k:>11s}" for k in BEHAVIOURS))
    for key in BEHAVIOURS:
        print(f"{key:>11s}" + "".join(f"{table[key]['scores'][k]:11.3f}" for k in BEHAVIOURS))


if __name__ == "__main__":
    main()
