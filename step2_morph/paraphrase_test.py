# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Compile unseen paraphrases of one prompt and grade the specs against the pre-registered rule.

Usage: python paraphrase_test.py paraphrase_right.json
Writes specs_para/<tag>_<k>.json and paraphrase_<tag>.json (one row per paraphrase).
"""

import json
import os
import sys

import compiler
from evolve_morph import violation

HERE = os.path.dirname(os.path.abspath(__file__))


def compile_one(text):
    spec, attempts = compiler.ask(text), []
    errs = compiler.validate(spec)
    while errs and len(attempts) < 2:
        attempts.append({"spec": spec, "errors": errs})
        spec = compiler.ask(text + "\nYour previous JSON was invalid: " + "; ".join(errs) +
                            ". Every morphology constraint needs descriptor, op and a numeric value.")
        errs = compiler.validate(spec)
    return spec, errs, attempts


def objective(spec, m):
    v = dict(m, abs_dx=abs(m["dx"]), abs_dy=abs(m["dy"]), abs_yaw=abs(m["yaw"]),
             travel=float((m["dx"] ** 2 + m["dy"] ** 2) ** 0.5))
    return sum(t["weight"] * v[t["descriptor"]] for t in spec["behaviour"])


def admits_legs(spec, n):
    return all(violation(n, c["op"], c["value"]) < 1e-9 for c in spec["morphology"] if c["descriptor"] == "n_legs")


def grade(spec, test):
    scores = {k: objective(spec, m) for k, m in test["primitives"].items()}
    best = max(scores, key=scores.get)
    beh_ok = best == "right" and sorted(scores.values())[-2] < scores["right"]
    has_legs = any(c["descriptor"] == "n_legs" for c in spec["morphology"])
    body_ok = has_legs and admits_legs(spec, 4) and not admits_legs(spec, 2) and not admits_legs(spec, 6)
    straight = objective(spec, test["primitives"]["right"]) > objective(spec, test["diagnostic"]["right_turning"])
    return body_ok, beh_ok, best, straight


def main(path):
    test = json.load(open(path, encoding="utf-8"))
    tag = os.path.splitext(os.path.basename(path))[0].replace("paraphrase_", "")
    os.makedirs(os.path.join(HERE, "specs_para"), exist_ok=True)
    rows = []
    for k, p in enumerate([{"kind": "original", "text": test["original"]}] + test["paraphrases"]):
        spec, errs, attempts = compile_one(p["text"])
        body_ok, beh_ok, best, straight = grade(spec, test) if not errs else (False, False, None, False)
        name = f"{tag}_{k:02d}"
        spec = {"name": name, "prompt": p["text"], "kind": p["kind"], "compiler": compiler.MODEL, **spec,
                "validation_errors": errs, "repair_attempts": attempts}
        json.dump(spec, open(os.path.join(HERE, "specs_para", name + ".json"), "w", encoding="utf-8"), indent=1,
                  ensure_ascii=False)
        beh = " ".join(f"{t['weight']:+g}*{t['descriptor']}" for t in spec["behaviour"])
        mor = ", ".join(f"{c['descriptor']} {c['op']} {c['value']:g}" for c in spec["morphology"])
        rows.append({"name": name, "kind": p["kind"], "text": p["text"], "objective": beh, "body": mor,
                     "body_ok": body_ok, "behaviour_ok": beh_ok, "preferred_primitive": best,
                     "prefers_straight": straight, "errors": errs})
        print(f"{name} [{p['kind']:10s}] {'OK ' if body_ok and beh_ok else 'BAD'} body={body_ok!s:5s} "
              f"beh={beh_ok!s:5s} straight={straight!s:5s} | {beh} | {mor}\n      \"{p['text']}\"", flush=True)
    json.dump(rows, open(os.path.join(HERE, f"paraphrase_{tag}_results.json"), "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    n = len(rows) - 1
    ok = sum(r["body_ok"] and r["behaviour_ok"] for r in rows[1:])
    print(f"\nparaphrases passing both: {ok}/{n}")


if __name__ == "__main__":
    main(sys.argv[1])
