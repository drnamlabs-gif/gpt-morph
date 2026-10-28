# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Does the compiler read the body from the text, or default to 4 legs?

Usage: python body_test.py paraphrase_body.json  -> specs_para/body_<k>.json, paraphrase_body_results.json
"""

import json
import os
import sys

from paraphrase_test import admits_legs, compile_one

HERE = os.path.dirname(os.path.abspath(__file__))
OPS = {"min": (">=", "=="), "max": ("<=", "==")}


def grade(spec, item):
    if "legs" not in item:
        return None
    legs_ok = [n for n in (2, 4, 6) if admits_legs(spec, n)] == sorted(item["legs"])
    need_ok = all(any(c["descriptor"] == d and c["op"] in OPS[way] for c in spec["morphology"])
                  for d, way in item.get("need", {}).items())
    return legs_ok and need_ok


def main(path):
    items = json.load(open(path, encoding="utf-8"))["items"]
    rows = []
    for k, it in enumerate(items):
        spec, errs, attempts = compile_one(it["text"])
        ok = grade(spec, it) if not errs else False
        right = [t["weight"] for t in spec["behaviour"] if t["descriptor"] == "dy"]
        json.dump({"name": f"body_{k:02d}", "prompt": it["text"], **spec, "validation_errors": errs,
                   "repair_attempts": attempts},
                  open(os.path.join(HERE, "specs_para", f"body_{k:02d}.json"), "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False)
        mor = ", ".join(f"{c['descriptor']} {c['op']} {c['value']:g}" for c in spec["morphology"]) or "(none)"
        beh = " ".join(f"{t['weight']:+g}*{t['descriptor']}" for t in spec["behaviour"])
        rows.append({"kind": it["kind"], "text": it["text"], "body": mor, "objective": beh, "body_ok": ok,
                     "dy_weight": right, "reasoning": spec.get("reasoning")})
        tag = "probe" if ok is None else ("OK " if ok else "BAD")
        print(f"{k:02d} [{it['kind']:14s}] {tag} | body: {mor} | {beh}\n     \"{it['text']}\"", flush=True)
    json.dump(rows, open(os.path.join(HERE, "paraphrase_body_results.json"), "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    scored = [r for r in rows if r["body_ok"] is not None]
    print(f"\nbody correct: {sum(r['body_ok'] for r in scored)}/{len(scored)}")


if __name__ == "__main__":
    main(sys.argv[1])
