# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Training data for the LoRA compiler: (sentence, canonical spec JSON).

Sentences = the student's templated set + paraphrases written by the local 27B LLM for each
(direction, body) class. Labels come from the class we asked for, never from an LLM-written spec.
Filters: test-only words (student.BANNED) and cosine > MAX_SIM to any test sentence are dropped,
so all 23 scored test sentences stay unseen. English only: Vietnamese/French tests stay cross-lingual.

Usage: python lora_data.py      -> lora/train.jsonl
"""

import itertools
import json
import os
import random
import re
import urllib.request

import numpy as np
from sentence_transformers import SentenceTransformer

import compiler
import student

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "lora")
MAX_SIM, PER_CLASS = 0.85, 12
DIR_MEANING = {"forward": "walks forward (straight ahead)", "backward": "walks backward (in reverse)",
               "left": "moves sideways toward its own left side, without turning",
               "right": "moves sideways toward its own right side, without turning",
               "turn_left": "turns on the spot to its left (counter-clockwise)",
               "turn_right": "turns on the spot to its right (clockwise)", "hop": "hops / jumps off the ground"}
BODY_MEANING = {"none": "a robot (say NOTHING about its body or legs)", "legs2": "a robot with exactly two legs",
                "legs4": "a robot with exactly four legs", "legs6": "a robot with exactly six legs",
                "long": "a robot with long legs", "short": "a robot with short legs",
                "light": "a robot that must be as light as possible"}


def spec_json(d, b):
    return json.dumps(student.to_spec(student.DK.index(d), student.BK.index(b)), separators=(",", ":"))


def generate(d, b):
    prompt = (f"Write {PER_CLASS} different requests a person might type to ask for {BODY_MEANING[b]} that "
              f"{DIR_MEANING[d]}. Vary the wording and sentence structure a lot (short commands, full sentences, "
              f"casual, technical). English only. Reply as JSON: {{\"requests\": [...]}}")
    body = json.dumps({"model": compiler.MODEL, "prompt": prompt, "stream": False, "think": False,
                       "format": {"type": "object", "required": ["requests"],
                                  "properties": {"requests": {"type": "array", "items": {"type": "string"}}}},
                       "options": {"temperature": 0.9, "seed": 0}}).encode()
    req = urllib.request.Request(compiler.OLLAMA, data=body, headers={"Content-Type": "application/json"})
    return json.loads(json.loads(urllib.request.urlopen(req, timeout=600).read())["response"])["requests"]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = [(s, student.DK[d], student.BK[b]) for s, d, b in student.training_set(random.Random(0))]
    for d, b in itertools.product(student.DK, student.BK):
        got = generate(d, b)
        rows += [(s.strip(), d, b) for s in got if s.strip()]
        print(f"{d:10s} {b:6s} +{len(got)}", flush=True)
    tests = [p["text"] for p in json.load(open(os.path.join(HERE, "paraphrase_right.json"), encoding="utf-8"))["paraphrases"]]
    tests += [it["text"] for it in json.load(open(os.path.join(HERE, "paraphrase_body.json"), encoding="utf-8"))["items"]]
    tests.append(json.load(open(os.path.join(HERE, "paraphrase_right.json"), encoding="utf-8"))["original"])
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    E = sbert.encode([r[0] for r in rows], normalize_embeddings=True)
    T = sbert.encode(tests, normalize_embeddings=True)
    sim = (E @ T.T).max(1)
    keep, n_ban, n_sim = [], 0, 0
    for r, s in zip(rows, sim):
        if any(re.search(r"\b" + w, r[0].lower()) for w in student.BANNED):
            n_ban += 1
        elif s > MAX_SIM:
            n_sim += 1
        else:
            keep.append(r)
    keep = list(dict.fromkeys(keep))
    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for s, d, b in keep:
            f.write(json.dumps({"text": s, "direction": d, "body": b, "spec": spec_json(d, b)}, ensure_ascii=False) + "\n")
    print(f"\n{len(rows)} sentences, dropped {n_ban} (test-only words) + {n_sim} (cos > {MAX_SIM} to a test "
          f"sentence), kept {len(keep)}; max kept similarity {max(s for r, s in zip(rows, sim) if r in keep):.2f}")


if __name__ == "__main__":
    main()
