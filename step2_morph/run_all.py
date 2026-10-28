# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Run every compiled spec x 3 seeds, plus the Text2Robot-style baseline.

Baseline (T2R): same morphology constraints as each B spec, but the behaviour
objective is fixed to forward speed (10 * dx), i.e. language shapes the body
only and a fixed velocity reward drives control, as in Text2Robot.
Skips runs whose summary.json already exists, so it can be restarted.
"""

import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import evolve_morph as E

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS, GENERATIONS = [0, 1, 2], 150


def jobs():
    specs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(HERE, "specs", "*.json")))]
    out = []
    for s in specs:
        out.append(s)
        if s["name"].startswith("B_"):
            out.append({**s, "name": s["name"] + "__T2R", "behaviour": [{"descriptor": "dx", "weight": 10.0}],
                        "baseline": "Text2Robot-style: language -> body constraints only; fixed forward-speed reward"})
    return out


def main():
    with ProcessPoolExecutor(E.N_WORKERS) as pool:
        for spec in jobs():
            for seed in SEEDS:
                if os.path.exists(os.path.join(HERE, "results", f"{spec['name']}_s{seed}", "summary.json")):
                    continue
                best, hist, wall = E.run(spec, seed, GENERATIONS, pool, log=lambda m: print(m, flush=True))
                print("saved", E.save(spec, seed, GENERATIONS, best, hist, wall), flush=True)


if __name__ == "__main__":
    main()
