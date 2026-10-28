# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Grade the LoRA predictions with the same rules as the LLM and the student; print all side by side.

Usage: python lora_grade.py      -> lora/graded.json
"""

import json
import os

import body_test
import paraphrase_test

HERE = os.path.dirname(os.path.abspath(__file__))


def parse(txt):
    try:
        s = json.loads(txt.strip())
        assert isinstance(s.get("behaviour"), list) and isinstance(s.get("morphology"), list)
        return s
    except Exception:
        return None


def right_ok(spec):
    """Direction right: dominant term is -dy and no abs_dy reward (same reading as the LLM tally)."""
    w = {t["descriptor"]: t["weight"] for t in spec["behaviour"]}
    return w.get("dy", 0) < 0 and w.get("abs_dy", 0) <= 0 and abs(w.get("dy", 0)) > max(
        [abs(v) for k, v in w.items() if k not in ("dy",)] + [0])


def main():
    P = json.load(open(os.path.join(HERE, "lora", "predictions.json"), encoding="utf-8"))
    right = json.load(open(os.path.join(HERE, "paraphrase_right.json"), encoding="utf-8"))
    items = json.load(open(os.path.join(HERE, "paraphrase_body.json"), encoding="utf-8"))["items"]
    llm_r = json.load(open(os.path.join(HERE, "paraphrase_right_results.json"), encoding="utf-8"))[1:]
    llm_b = json.load(open(os.path.join(HERE, "paraphrase_body_results.json"), encoding="utf-8"))
    stu = json.load(open(os.path.join(HERE, "student_results.json"), encoding="utf-8"))
    nr = len(right["paraphrases"])
    rows = []
    for k, text in enumerate(P["texts"]):
        r = {"text": text}
        for who in ("base", "lora"):
            s = parse(P[who][k])
            if k < nr:
                g = paraphrase_test.grade(s, right) if s else (False, False, None, False)
                r[who] = (g[0], g[1])
            else:
                it = items[k - nr]
                r[who] = (body_test.grade(s, it) if s else (False if "legs" in it else None),
                          right_ok(s) if s else False)
        if k < nr:
            r["kind"] = right["paraphrases"][k]["kind"]
            r["llm27b"] = (llm_r[k]["body_ok"], llm_r[k]["behaviour_ok"])
            r["student"] = (stu["right"][k]["body_ok"], stu["right"][k]["behaviour_ok"])
        else:
            j = k - nr
            r["kind"] = items[j]["kind"]
            r["llm27b"] = (llm_b[j]["body_ok"], stu["body"][j]["llm_dir_ok"])
            r["student"] = (stu["body"][j]["body_ok"], stu["body"][j]["dir_ok"])
        rows.append(r)

    f = lambda t: ("?" if t[0] is None else ("B" if t[0] else "-")) + ("D" if t[1] else "-")
    who = ["llm27b", "student", "base", "lora"]
    print(f"{'kind':15s} {'27B 0-shot':11s}{'SBERT+MLP':11s}{'1.5B base':11s}{'1.5B LoRA':11s}| sentence")
    for i, r in enumerate(rows):
        if i == nr:
            print("-" * 60)
        print(f"{r['kind']:15s} " + "".join(f"{f(r[w]):11s}" for w in who) + f"| {r['text']}")
    print()
    for name, part in (("right paraphrases (11)", rows[:nr]), ("body variants (12 scored)", rows[nr:])):
        sc = [r for r in part if r["llm27b"][0] is not None]
        print(name)
        for w in who:
            print(f"   {w:8s} body {sum(bool(r[w][0]) for r in sc):2d}/{len(sc)}   direction "
                  f"{sum(bool(r[w][1]) for r in sc):2d}/{len(sc)}   both {sum(bool(r[w][0]) and bool(r[w][1]) for r in sc):2d}/{len(sc)}")
    json.dump(rows, open(os.path.join(HERE, "lora", "graded.json"), "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)


if __name__ == "__main__":
    main()
