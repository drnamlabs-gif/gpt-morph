# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""P2I student: learn prompt embedding -> CPG genome, then test on unseen paraphrases.

Specialists = the 5 evolved champions (one per behaviour). Student = a small
MLP conditioned on the frozen SBERT embedding of the prompt, trained by
regression onto the champion genome of the prompt's behaviour, using 12
training paraphrases per behaviour (train_paraphrases.json, disjoint from the
fixed test set). This is the specialist -> student distillation with
paraphrase augmentation used in the BETSE language->intervention work.

Three mappings are compared on the same pre-registered test set
(test_prompts.json), each scored by RUNNING the output genome in MuJoCo and
reading the executed behaviour off the motion (test_paraphrase.executed_behaviour):
  R1   retrieval, 1 training prompt per behaviour (the earlier baseline)
  R12  retrieval, nearest of the 12 training paraphrases
  MLP  student network regressing the 25 genes (5 training seeds)
  ROUTER  student network with a softmax over the 5 champions, trained with
          cross-entropy; the chosen champion is run (5 training seeds)
Gaits are fragile in genome space (champion + N(0, 0.02) noise changes the
behaviour 15% of the time), so a regressed genome that lands between
champions tends to break; the router selects instead of blending.
Outputs: results/p2i_student.json, doc/figs/fig_p2i_student.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import frog
from test_paraphrase import KEYS, executed_behaviour

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEEDS = [0, 1, 2, 3, 4]
HIDDEN, EPOCHS, LR, WD, DROPOUT = 64, 1500, 1e-3, 1e-4, 0.2


class Student(torch.nn.Module):
    def __init__(self, d_in, n_out):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d_in, HIDDEN), torch.nn.ReLU(), torch.nn.Dropout(DROPOUT),
                                       torch.nn.Linear(HIDDEN, n_out), torch.nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


def train_student(X, Y, seed):
    torch.manual_seed(seed)
    model = Student(X.shape[1], Y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(X), Y)
        loss.backward()
        opt.step()
    model.eval()
    return model, float(loss.detach())


def train_router(X, labels, seed):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(X.shape[1], HIDDEN), torch.nn.ReLU(), torch.nn.Dropout(DROPOUT),
                                torch.nn.Linear(HIDDEN, len(KEYS)))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor([KEYS.index(k) for k in labels])
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
    model.eval()
    return model, float(loss.detach())


def main():
    test_spec = json.load(open(os.path.join(HERE, "test_prompts.json"), encoding="utf-8"))
    train_spec = json.load(open(os.path.join(HERE, "train_paraphrases.json"), encoding="utf-8"))
    test_texts = [t["text"] for t in test_spec["test"]]
    probe_texts = [p["text"] for p in test_spec["probe"]]
    train_texts, train_labels = [], []
    for k in KEYS:
        for s in train_spec[k]:
            train_texts.append(s)
            train_labels.append(k)
    overlap = set(s.lower().strip() for s in train_texts) & set(s.lower().strip() for s in test_texts + probe_texts)
    assert not overlap, overlap

    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    enc = lambda xs: sbert.encode(xs, normalize_embeddings=True)
    E_tr, E_te, E_pr = enc(train_texts), enc(test_texts), enc(probe_texts)
    E_r1 = enc([train_spec[k][0] for k in KEYS])
    genomes = {k: np.array(json.load(open(os.path.join(RES, k, "best_genome.json")))) for k in KEYS}
    own = {k: json.load(open(os.path.join(RES, k, "summary.json")))["fitness"] for k in KEYS}
    G = np.stack([genomes[k] for k in KEYS])

    def execute(genome):
        d = frog.rollout(genome)
        return executed_behaviour(d, own)[0], d

    def score(name, genome_for):
        rows = []
        for t, e in zip(test_spec["test"], E_te):
            g = genome_for(e)
            ex, d = execute(g)
            rows.append({"text": t["text"], "label": t["label"], "kind": t["kind"], "executed": ex,
                         "correct": ex == t["label"], "dist_to_nearest_champion": float(np.min(np.linalg.norm(G - g, axis=1)))})
        probes = []
        for p, e in zip(test_spec["probe"], E_pr):
            ex, d = execute(genome_for(e))
            probes.append({"text": p["text"], "intended": p["intended"], "executed": ex,
                           "dx": d["dx"], "dy": d["dy"], "yaw_deg": float(np.degrees(d["yaw"]))})
        acc = {kind: float(np.mean([r["correct"] for r in rows if r["kind"] == kind])) for kind in ("near", "far")}
        acc["all"] = float(np.mean([r["correct"] for r in rows]))
        print(f"{name:10s} all {acc['all']:.2f}  near {acc['near']:.2f}  far {acc['far']:.2f}", flush=True)
        return {"accuracy": acc, "tests": rows, "probes": probes}

    out = {"train_paraphrases_per_behaviour": len(train_spec[KEYS[0]]), "n_test": len(test_texts)}
    out["R1"] = score("R1", lambda e: genomes[KEYS[int(np.argmax(E_r1 @ e))]])
    out["R12"] = score("R12", lambda e: genomes[train_labels[int(np.argmax(E_tr @ e))]])
    Y = np.stack([genomes[k] for k in train_labels])
    out["MLP"] = []
    for s in SEEDS:
        model, loss = train_student(E_tr, Y, s)
        with torch.no_grad():
            f = lambda e, m=model: m(torch.tensor(e[None], dtype=torch.float32))[0].numpy().astype(float)
            r = score(f"MLP s{s}", f)
        r["seed"], r["final_train_mse"] = s, loss
        out["MLP"].append(r)
    out["ROUTER"] = []
    for s in SEEDS:
        model, loss = train_router(E_tr, train_labels, s)
        with torch.no_grad():
            f = lambda e, m=model: genomes[KEYS[int(m(torch.tensor(e[None], dtype=torch.float32))[0].argmax())]]
            r = score(f"ROUTER s{s}", f)
        r["seed"], r["final_train_ce"] = s, loss
        out["ROUTER"].append(r)
    rt = np.array([[r["accuracy"][k] for k in ("all", "near", "far")] for r in out["ROUTER"]])
    out["ROUTER_summary"] = {"mean": dict(zip(("all", "near", "far"), rt.mean(0).tolist())),
                             "sd": dict(zip(("all", "near", "far"), rt.std(0, ddof=1).tolist()))}
    print("ROUTER mean", out["ROUTER_summary"])
    mlp = np.array([[r["accuracy"][k] for k in ("all", "near", "far")] for r in out["MLP"]])
    out["MLP_summary"] = {"mean": dict(zip(("all", "near", "far"), mlp.mean(0).tolist())),
                          "sd": dict(zip(("all", "near", "far"), mlp.std(0, ddof=1).tolist())),
                          "median_dist_to_champion": float(np.median([t["dist_to_nearest_champion"]
                                                                      for r in out["MLP"] for t in r["tests"]]))}
    json.dump(out, open(os.path.join(RES, "p2i_student.json"), "w"), indent=1)
    print("MLP mean", out["MLP_summary"])

    # figure: accuracy by method, and per-prompt correctness grid
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw={"width_ratios": [1, 1.6]})
    names = ["R1\n(1 prompt,\nretrieval)", "R12\n(12 paraphrases,\nretrieval)",
             "MLP\n(regress genes,\n5 seeds)", "Router\n(choose gait,\n5 seeds)"]
    x = np.arange(4)
    for j, (kind, col) in enumerate([("near", "#8FC9A1"), ("far", "#2E8B57"), ("all", "#16213A")]):
        vals = [out["R1"]["accuracy"][kind], out["R12"]["accuracy"][kind], out["MLP_summary"]["mean"][kind],
                out["ROUTER_summary"]["mean"][kind]]
        err = [0, 0, out["MLP_summary"]["sd"][kind], out["ROUTER_summary"]["sd"][kind]]
        a.bar(x + (j - 1) * 0.26, vals, 0.26, yerr=err, color=col, label=kind, capsize=3)
    a.axhline(0.2, ls=":", color="#999")
    a.text(3.3, 0.21, "chance", fontsize=8, color="#777")
    a.set_xticks(x, names, fontsize=9)
    a.set_ylim(0, 1.08)
    a.set_ylabel("fraction of unseen prompts with the right gait")
    a.set_title("Unseen paraphrases, executed in MuJoCo", fontsize=10)
    a.legend(fontsize=8)

    cols = ["R1", "R12"] + [f"MLP s{s}" for s in SEEDS] + [f"Rout s{s}" for s in SEEDS]
    grid = np.array([[r["correct"] for r in out["R1"]["tests"]], [r["correct"] for r in out["R12"]["tests"]]]
                    + [[t["correct"] for t in r["tests"]] for r in out["MLP"]]
                    + [[t["correct"] for t in r["tests"]] for r in out["ROUTER"]], float).T
    b.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    b.set_xticks(range(len(cols)), cols, fontsize=8)
    b.set_yticks(range(len(test_texts)), [f"{t['label']}/{t['kind']}: {t['text']}" for t in test_spec["test"]], fontsize=7)
    b.set_title("Per prompt (green = right gait)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "doc", "figs", "fig_p2i_student.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
