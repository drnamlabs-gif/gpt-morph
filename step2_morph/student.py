# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Embedding student for step 2: sentence -> spec, trained, then tested on the same unseen sentences as the LLM.

frozen SBERT (all-MiniLM-L6-v2, 384-d) -> one shared MLP trunk -> two heads
  direction head: forward / backward / left / right / turn_left / turn_right / hop
  body head:      none / 2 legs / 4 legs / 6 legs / long legs / short legs / light
Each class maps to one canonical spec fragment (CANON), so the output is a spec JSON of the same form
the LLM writes. Training sentences are templated paraphrases; words that occur only in the test files
(BANNED) are never used, so every test sentence is unseen and the rare ones are lexically new.

Usage: python student.py      -> student_results.json, printed side-by-side with the LLM
"""

import itertools
import json
import os
import random
import re

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import body_test
import paraphrase_test

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0
torch.set_num_threads(8)

DIRS = {
    "forward": ["walks forward", "moves forward", "goes straight ahead", "walks ahead", "travels forward",
                "steps forward", "advances", "walks toward the front"],
    "backward": ["walks backward", "moves backwards", "goes in reverse", "backs up", "walks toward the rear",
                 "steps backward", "retreats", "travels backward"],
    "left": ["walks sideways to the left", "moves to the left side", "steps sideways left", "slides left",
             "travels to its left", "goes left sideways", "sidesteps to the left", "walks leftward"],
    "right": ["walks sideways to the right", "moves to the right side", "steps sideways right", "slides right",
              "travels to its right", "goes right sideways", "sidesteps to the right", "walks rightward"],
    "turn_left": ["turns left on the spot", "spins counterclockwise", "rotates left in place",
                  "turns around to the left", "pivots left without moving", "spins to the left"],
    "turn_right": ["turns right on the spot", "spins clockwise", "rotates right in place",
                   "turns around to the right", "pivots right without moving", "spins to the right"],
    "hop": ["hops", "jumps", "bounces off the ground", "leaps", "hops in place", "jumps up and down"],
}
BODIES = {
    "none": ["a robot", "a legged robot", "a walking robot", "a small robot"],
    "legs2": ["a two-legged robot", "a robot with two legs", "a 2-legged robot", "a robot on two legs"],
    "legs4": ["a four-legged robot", "a robot with four legs", "a 4-legged robot", "a robot on four legs"],
    "legs6": ["a six-legged robot", "a robot with six legs", "a 6-legged robot", "a robot on six legs"],
    "long": ["a long-legged robot", "a robot with tall legs", "a robot standing on long legs",
             "a robot with lengthy legs"],
    "short": ["a short-legged robot", "a robot with short legs", "a squat robot with tiny legs",
              "a robot that sits close to the ground"],
    "light": ["a lightweight robot", "a robot that weighs very little", "a featherweight robot",
              "a light-bodied robot"],
}
TEMPLATES = ["{b} that {d}", "{b} which {d}", "{B}. It {d}.", "I want {b} that {d}", "Build {b} that {d}"]
BANNED = ["strafe", "starboard", "hexapod", "biped", "insect", "crab", "quadruped", "pair", "stubby", "spider",
          "painted", "three", "eight", "flat", "right-hand", "facing"]
CANON_DIR = {"forward": [("dx", 10)], "backward": [("dx", -10)], "left": [("dy", 10)], "right": [("dy", -10)],
             "turn_left": [("yaw", 10), ("travel", -5)], "turn_right": [("yaw", -10), ("travel", -5)],
             "hop": [("airtime", 10)]}
CANON_BODY = {"none": [], "legs2": [("n_legs", "==", 2)], "legs4": [("n_legs", "==", 4)],
              "legs6": [("n_legs", "==", 6)], "long": [("mean_leg_len", ">=", 0.35)],
              "short": [("mean_leg_len", "<=", 0.15)], "light": [("mass", "<=", 1.2)]}
DK, BK = list(DIRS), list(BODIES)


def training_set(rng):
    rows = []
    for d, b in itertools.product(DK, BK):
        combos = [(t, dp, bp) for t in TEMPLATES for dp in DIRS[d] for bp in BODIES[b]]
        for t, dp, bp in rng.sample(combos, 6):
            s = t.format(b=bp, B=bp[0].upper() + bp[1:], d=dp)
            assert not any(re.search(r"\b" + w, s.lower()) for w in BANNED), s
            rows.append((s, DK.index(d), BK.index(b)))
    return rows


def to_spec(d, b):
    return {"behaviour": [{"descriptor": k, "weight": float(w)} for k, w in CANON_DIR[DK[d]]],
            "morphology": [{"descriptor": k, "op": o, "value": float(v)} for k, o, v in CANON_BODY[BK[b]]]}


class Student(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Sequential(torch.nn.Linear(384, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2))
        self.dir, self.body = torch.nn.Linear(128, len(DK)), torch.nn.Linear(128, len(BK))

    def forward(self, x):
        h = self.trunk(x)
        return self.dir(h), self.body(h)


def main():
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    rows = training_set(rng)
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X = torch.tensor(sbert.encode([r[0] for r in rows], normalize_embeddings=True))
    yd, yb = torch.tensor([r[1] for r in rows]), torch.tensor([r[2] for r in rows])
    net = Student()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(400):
        net.train()
        ld, lb = net(X)
        loss = torch.nn.functional.cross_entropy(ld, yd) + torch.nn.functional.cross_entropy(lb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        ld, lb = net(X)
    print(f"train: {len(rows)} sentences, dir acc {(ld.argmax(1) == yd).float().mean():.2f}, "
          f"body acc {(lb.argmax(1) == yb).float().mean():.2f}")

    def predict(texts):
        with torch.no_grad():
            a, b = net(torch.tensor(sbert.encode(texts, normalize_embeddings=True)))
        return a.argmax(1).tolist(), b.argmax(1).tolist()

    out = {}
    right = json.load(open(os.path.join(HERE, "paraphrase_right.json"), encoding="utf-8"))
    llm_r = json.load(open(os.path.join(HERE, "paraphrase_right_results.json"), encoding="utf-8"))[1:]
    texts = [p["text"] for p in right["paraphrases"]]
    pd_, pb = predict(texts)
    print("\n== paraphrases of 'a four-legged robot that walks sideways to the right' ==")
    print(f"{'kind':11s} {'LLM':9s} {'student':9s} student prediction   | sentence")
    res = []
    for p, d, b, l in zip(right["paraphrases"], pd_, pb, llm_r):
        body_ok, beh_ok, _, _ = paraphrase_test.grade(to_spec(d, b), right)
        res.append({"kind": p["kind"], "text": p["text"], "dir": DK[d], "body": BK[b], "body_ok": body_ok,
                    "behaviour_ok": beh_ok, "llm_body_ok": l["body_ok"], "llm_behaviour_ok": l["behaviour_ok"]})
        f = lambda x, y: ("B" if x else "-") + ("D" if y else "-")
        print(f"{p['kind']:11s} {f(l['body_ok'], l['behaviour_ok']):9s} {f(body_ok, beh_ok):9s} "
              f"{DK[d] + ' / ' + BK[b]:20s} | {p['text']}")
    out["right"] = res

    items = json.load(open(os.path.join(HERE, "paraphrase_body.json"), encoding="utf-8"))["items"]
    llm_b = json.load(open(os.path.join(HERE, "paraphrase_body_results.json"), encoding="utf-8"))
    pd_, pb = predict([it["text"] for it in items])
    print("\n== same behaviour (right), different bodies ==")
    print(f"{'kind':15s} {'LLM':9s} {'student':9s} student prediction   | sentence")
    res = []
    for it, d, b, l in zip(items, pd_, pb, llm_b):
        ok = body_test.grade(to_spec(d, b), it)
        llm_dir = bool(l["dy_weight"]) and l["dy_weight"][0] < 0 and "abs_dy" not in l["objective"]
        res.append({"kind": it["kind"], "text": it["text"], "dir": DK[d], "body": BK[b], "body_ok": ok,
                    "dir_ok": DK[d] == "right", "llm_body_ok": l["body_ok"], "llm_dir_ok": llm_dir})
        f = lambda x, y: ("B" if x else ("?" if x is None else "-")) + ("D" if y else "-")
        print(f"{it['kind']:15s} {f(l['body_ok'], llm_dir):9s} {f(ok, DK[d] == 'right'):9s} "
              f"{DK[d] + ' / ' + BK[b]:20s} | {it['text']}")
    out["body"] = res

    def tally(rs, bk, dk):
        sc = [r for r in rs if r[bk] is not None]
        return sum(r[bk] for r in sc), sum(r[dk] for r in sc), len(sc)
    for name, rs, keys in (("right", out["right"], [("llm_body_ok", "llm_behaviour_ok"), ("body_ok", "behaviour_ok")]),
                           ("body", out["body"], [("llm_body_ok", "llm_dir_ok"), ("body_ok", "dir_ok")])):
        for who, (bk, dk) in zip(("LLM", "student"), keys):
            b, d, n = tally(rs, bk, dk)
            print(f"{name:6s} {who:8s} body {b}/{n}  direction {d}/{n}")
    json.dump(out, open(os.path.join(HERE, "student_results.json"), "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)


if __name__ == "__main__":
    main()
