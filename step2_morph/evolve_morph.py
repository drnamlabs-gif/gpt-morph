# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Co-evolve body + CPG for a compiled language spec.

A spec (what the language compiler produces) is JSON:
  {"prompt": "...",
   "behaviour":  [{"descriptor": "dx", "weight": 1.0}, ...],          maximise sum
   "morphology": [{"descriptor": "n_legs", "op": "==", "value": 6}, ...]}
Behaviour descriptors: dx, dy, yaw, airtime, abs_dx, abs_dy, abs_yaw, travel.
Morphology descriptors: n_legs, n_joints, mean_leg_len, max_leg_len,
  torso_L, torso_W, mass, stand_z.
Fitness = sum(weight * descriptor) - 2 * [flipped] - sum over violated
constraints of (2 + 10 * relative violation). Same GA as step 1.

Usage: python evolve_morph.py <spec.json> <seed> [generations]
Writes results/<spec name>_s<seed>/{best_genome.json,summary.json,history.json}
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import morph

HERE = os.path.dirname(os.path.abspath(__file__))
POP, ELITE, TOURN, MUT_RATE, MUT_SIGMA = 64, 6, 3, 0.15, 0.15
FLIP_PENALTY = 2.0
N_WORKERS = int(os.environ.get("N_WORKERS", "12"))
BEHAVIOUR_DESC = ["dx", "dy", "yaw", "airtime", "abs_dx", "abs_dy", "abs_yaw", "travel"]
MORPH_DESC = ["n_legs", "n_joints", "mean_leg_len", "max_leg_len", "torso_L", "torso_W", "mass", "stand_z"]


def behaviour_values(d):
    v = {k: d[k] for k in ("dx", "dy", "yaw", "airtime")}
    v.update(abs_dx=abs(d["dx"]), abs_dy=abs(d["dy"]), abs_yaw=abs(d["yaw"]), travel=float(np.hypot(d["dx"], d["dy"])))
    return v


def violation(value, op, target):
    scale = max(abs(target), 1e-3)
    if op == "==":
        return abs(value - target) / scale
    if op == ">=":
        return max(0.0, target - value) / scale
    if op == "<=":
        return max(0.0, value - target) / scale
    raise ValueError(op)


def evaluate(spec, d):
    v = behaviour_values(d)
    beh = sum(t["weight"] * v[t["descriptor"]] for t in spec["behaviour"])
    pen = 0.0
    viol = []
    for c in spec.get("morphology", []):
        x = violation(d[c["descriptor"]], c["op"], c["value"])
        if x > 1e-9:
            pen += 2.0 + 10.0 * x
            viol.append(c["descriptor"])
    if d["flipped_at"] is not None:
        pen += FLIP_PENALTY
    return beh - pen, beh, viol


def _eval(genome):
    return [morph.rollout(genome, **c) for c in morph.CONDITIONS]


def robust(spec, descs):
    """Worst case over the friction conditions; descriptors of the nominal one."""
    scored = [evaluate(spec, d) for d in descs]
    k = int(np.argmin([s[0] for s in scored]))
    return scored[k][0], scored[0][1], scored[0][2]


def run(spec, seed, generations, pool, log=print):
    rng = np.random.default_rng(seed)
    pop = rng.random((POP, morph.N_GENES))
    best, history, t0 = (-np.inf, None, None), [], time.time()
    for gen in range(generations):
        descs = list(pool.map(_eval, pop, chunksize=2))
        scored = [robust(spec, d) for d in descs]
        fit = np.array([s[0] for s in scored])
        descs = [d[0] for d in descs]
        order = np.argsort(fit)[::-1]
        pop, fit, descs = pop[order], fit[order], [descs[i] for i in order]
        if fit[0] > best[0]:
            best = (float(fit[0]), pop[0].copy(), descs[0])
        d = descs[0]
        history.append({"generation": gen, "best": float(fit[0]), "mean": float(fit.mean()),
                        "n_legs": d["n_legs"], "mean_leg_len": d["mean_leg_len"], "n_joints": d["n_joints"]})
        if gen % 20 == 0 or gen == generations - 1:
            log(f"  [{spec['name']} s{seed}] gen {gen:3d} best {fit[0]:7.3f} mean {fit.mean():7.3f} "
                f"legs {d['n_legs']} joints {d['n_joints']} leg {d['mean_leg_len']:.2f} {time.time() - t0:4.0f}s")
        children = [pop[i].copy() for i in range(ELITE)]
        while len(children) < POP:
            a = pop[min(rng.integers(0, POP, TOURN))]
            b = pop[min(rng.integers(0, POP, TOURN))]
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            c = rng.uniform(lo - 0.5 * (hi - lo), hi + 0.5 * (hi - lo))
            m = rng.random(morph.N_GENES) < MUT_RATE
            c[m] += rng.normal(0, MUT_SIGMA, m.sum())
            children.append(np.clip(c, 0, 1))
        pop = np.array(children)
    return best, history, time.time() - t0


def save(spec, seed, generations, best, history, wall):
    fit, genome, desc = best
    out = os.path.join(HERE, "results", f"{spec['name']}_s{seed}")
    os.makedirs(out, exist_ok=True)
    total, beh, viol = evaluate(spec, desc)
    json.dump(genome.tolist(), open(os.path.join(out, "best_genome.json"), "w"))
    json.dump(history, open(os.path.join(out, "history.json"), "w"))
    json.dump({"spec": spec, "seed": seed, "generations": generations, "pop": POP, "fitness": total,
               "behaviour_score": beh, "violated": viol, "descriptors": desc, "body": morph.decode_body(genome),
               "evaluations": POP * generations, "wall_clock_s": wall},
              open(os.path.join(out, "summary.json"), "w"), indent=1)
    return out


def main():
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    seed = int(sys.argv[2])
    generations = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    morph.configure(spec.get("physics"))
    with ProcessPoolExecutor(N_WORKERS, initializer=morph.configure, initargs=(spec.get("physics"),)) as pool:
        best, history, wall = run(spec, seed, generations, pool)
    print("saved", save(spec, seed, generations, best, history, wall))


if __name__ == "__main__":
    main()
