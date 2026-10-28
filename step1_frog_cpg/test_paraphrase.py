# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Unseen-paraphrase test for the step-1 prompt -> gait mapping.

Training set = the 5 prompts in evolve.BEHAVIOURS and their evolved champions.
Mapping (ZapGPT retrieval): frozen SBERT all-MiniLM-L6-v2 embeds the prompt,
the nearest training prompt by cosine similarity picks the champion genome.
The genome is then run in MuJoCo and the executed behaviour is read off the
robot's motion, independently of the retrieval: each of the 5 metrics is
divided by the value its own champion reached, and the largest ratio names
the behaviour. A test prompt passes when that executed behaviour equals the
prompt's label (written in test_prompts.json before any test was run).

Probes (never-trained behaviours) are run and described, not scored.
Outputs: results/paraphrase_test.json, doc/figs/fig_paraphrase.png, gifs/test_*.gif
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

import evolve
import frog
import render

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
KEYS = list(evolve.BEHAVIOURS)


def executed_behaviour(desc, own):
    ratios = {k: evolve.BEHAVIOURS[k][1](desc) / own[k] for k in KEYS}
    return max(ratios, key=ratios.get), ratios


def main():
    spec = json.load(open(os.path.join(HERE, "test_prompts.json"), encoding="utf-8"))
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    train_text = [spec["train"][k] for k in KEYS]
    E_train = sbert.encode(train_text, normalize_embeddings=True)
    genomes = {k: np.array(json.load(open(os.path.join(RES, k, "best_genome.json")))) for k in KEYS}
    own = {k: json.load(open(os.path.join(RES, k, "summary.json")))["fitness"] for k in KEYS}

    def run(text):
        e = sbert.encode([text], normalize_embeddings=True)[0]
        sims = E_train @ e
        k = KEYS[int(np.argmax(sims))]
        desc = frog.rollout(genomes[k])
        ex, ratios = executed_behaviour(desc, own)
        return {"text": text, "retrieved": k, "cos": {KEYS[i]: float(sims[i]) for i in range(len(KEYS))},
                "cos_max": float(sims.max()), "executed": ex, "descriptors": desc,
                "ratios": {a: float(b) for a, b in ratios.items()}}

    tests = []
    for t in spec["test"]:
        r = run(t["text"])
        r.update(label=t["label"], kind=t["kind"], correct=r["executed"] == t["label"])
        tests.append(r)
        print(f"{'OK ' if r['correct'] else 'XX '} [{t['kind']:4s}] {t['label']:9s} <- {r['retrieved']:9s} "
              f"cos {r['cos_max']:.2f}  \"{t['text']}\"")
    probes = []
    for p in spec["probe"]:
        r = run(p["text"])
        r["intended"] = p["intended"]
        probes.append(r)
        d = r["descriptors"]
        print(f"PROBE {p['intended']:34s} -> {r['retrieved']:9s} cos {r['cos_max']:.2f} "
              f"dx {d['dx']:+.2f} dy {d['dy']:+.2f} yaw {np.degrees(d['yaw']):+.0f}")

    acc = {kind: float(np.mean([t["correct"] for t in tests if t["kind"] == kind])) for kind in ("near", "far")}
    acc["all"] = float(np.mean([t["correct"] for t in tests]))
    per_label = {k: float(np.mean([t["correct"] for t in tests if t["label"] == k])) for k in KEYS}
    out = {"model": "sentence-transformers/all-MiniLM-L6-v2", "mapping": "nearest training prompt (retrieval)",
           "accuracy": acc, "per_label": per_label, "tests": tests, "probes": probes}
    json.dump(out, open(os.path.join(RES, "paraphrase_test.json"), "w"), indent=1)
    print("accuracy", acc, "\nper label", per_label)

    # figure: cosine to nearest training prompt vs correctness, and confusion matrix
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
    order = sorted(range(len(tests)), key=lambda i: (KEYS.index(tests[i]["label"]), tests[i]["kind"]))
    y = np.arange(len(order))
    for yy, i in zip(y, order):
        t = tests[i]
        a.barh(yy, t["cos_max"], color="#2CA02C" if t["correct"] else "#D62728",
               alpha=1.0 if t["kind"] == "far" else 0.55)
        a.text(0.01, yy, f"{t['text']}", va="center", fontsize=7.5, color="white" if t["cos_max"] > 0.35 else "black")
        a.text(t["cos_max"] + 0.01, yy, f"→ {t['retrieved']}", va="center", fontsize=7)
    a.set_yticks(y, [f"{tests[i]['label']} ({tests[i]['kind']})" for i in order], fontsize=7.5)
    a.invert_yaxis()
    a.set_xlim(0, 1.15)
    a.set_xlabel("cosine similarity to the nearest training prompt (SBERT)")
    a.set_title(f"Unseen paraphrases: {100 * acc['all']:.0f}% correct "
                f"(near {100 * acc['near']:.0f}%, far {100 * acc['far']:.0f}%)\n"
                "green = executed gait matches the prompt, red = wrong gait", fontsize=10)
    C = np.zeros((len(KEYS), len(KEYS)))
    for t in tests:
        C[KEYS.index(t["label"]), KEYS.index(t["executed"])] += 1
    b.imshow(C, cmap="Greens")
    for i in range(len(KEYS)):
        for j in range(len(KEYS)):
            if C[i, j]:
                b.text(j, i, int(C[i, j]), ha="center", va="center", fontsize=11)
    b.set_xticks(range(len(KEYS)), KEYS, rotation=30, fontsize=8)
    b.set_yticks(range(len(KEYS)), KEYS, fontsize=8)
    b.set_xlabel("executed behaviour (measured)")
    b.set_ylabel("prompt label")
    b.set_title("Confusion (5 prompts per label)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "doc", "figs"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "doc", "figs", "fig_paraphrase.png"), dpi=150)
    plt.close(fig)

    # GIFs: every wrong test prompt, one far success per label, and every probe
    picks = [t for t in tests if not t["correct"]]
    for k in KEYS:
        ok = [t for t in tests if t["label"] == k and t["kind"] == "far" and t["correct"]]
        picks += ok[:1]
    picks += probes
    for n, t in enumerate(picks):
        d = t["descriptors"]
        tag = t.get("label", "probe")
        render.render(genomes[t["retrieved"]], os.path.join(HERE, "gifs", f"test_{n:02d}_{tag}.gif"),
                      [f"\"{t['text']}\"  ->  {t['retrieved']} (cos {t['cos_max']:.2f})",
                       f"dx {d['dx']:+.2f} m  dy {d['dy']:+.2f} m  yaw {np.degrees(d['yaw']):+.0f} deg"])


if __name__ == "__main__":
    main()
