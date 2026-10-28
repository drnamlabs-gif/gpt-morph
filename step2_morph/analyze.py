# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Step-2 analysis: does the robot do what the prompt says, and what body did it get?

For every run: body check and behaviour check from intent_checks.json
(human-written, independent of the compiler). Compares ours (LLM-compiled
behaviour + body) against the Text2Robot-style baseline (same body constraints,
fixed forward-speed reward). Also renders a body gallery and seed-0 GIFs.
Outputs: results/analysis.json, figs/*.png, gifs/*.gif
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import render

HERE = os.path.dirname(os.path.abspath(__file__))
CHK = json.load(open(os.path.join(HERE, "intent_checks.json"), encoding="utf-8"))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)


def check(prompt_name, d):
    env = {**d, "travel": float(np.hypot(d["dx"], d["dy"])), "abs": abs}
    rules = CHK["prompts"][prompt_name]
    body = all(eval(r, {}, env) for r in rules["body"])
    beh = all(eval(CHK["behaviour_rules"][b], {}, env) for b in rules["behaviour"]) and d["flipped_at"] is None
    return body, beh


def load_runs():
    runs = []
    for p in sorted(glob.glob(os.path.join(HERE, "results", "*_s*", "summary.json"))):
        s = json.load(open(p))
        name = s["spec"]["name"]
        base = name.replace("__T2R", "")
        body, beh = check(base, s["descriptors"])
        runs.append({"run": os.path.basename(os.path.dirname(p)), "name": name, "prompt_name": base,
                     "method": "T2R" if name.endswith("__T2R") else "ours", "seed": s["seed"],
                     "prompt": s["spec"]["prompt"], "body_ok": body, "behaviour_ok": beh, "both_ok": body and beh,
                     "d": s["descriptors"], "violated": s["violated"],
                     "genome": os.path.join(os.path.dirname(p), "best_genome.json")})
    return runs


def main():
    runs = load_runs()
    table = {}
    for r in runs:
        key = (r["prompt_name"], r["method"])
        table.setdefault(key, []).append(r)
    summary = {}
    print(f"{'prompt':17s} {'method':5s}  body  behav  both   (per seed: dx dy yaw air legs leglen standz)")
    for (pn, m), rs in sorted(table.items()):
        b = np.mean([x["body_ok"] for x in rs])
        h = np.mean([x["behaviour_ok"] for x in rs])
        both = np.mean([x["both_ok"] for x in rs])
        summary[f"{pn}|{m}"] = {"n": len(rs), "body": b, "behaviour": h, "both": both}
        det = " | ".join(f"{x['d']['dx']:+.1f} {x['d']['dy']:+.1f} {x['d']['yaw']:+.0f} {x['d']['airtime']:.2f} "
                         f"{x['d']['n_legs']} {x['d']['mean_leg_len']:.2f} {x['d']['stand_z']:.2f}" for x in rs)
        print(f"{pn:17s} {m:5s}  {b:.2f}  {h:.2f}  {both:.2f}   {det}")
    agg = {}
    for m in ("ours", "T2R"):
        rs = [r for r in runs if r["method"] == m and r["prompt_name"].startswith("B_")]
        if rs:
            agg[m] = {k: float(np.mean([r[k] for r in rs])) for k in ("body_ok", "behaviour_ok", "both_ok")}
            agg[m]["n_runs"] = len(rs)
    rs = [r for r in runs if r["prompt_name"].startswith("A_")]
    agg["A_ours"] = {k: float(np.mean([r[k] for r in rs])) for k in ("behaviour_ok",)}
    print("B prompts:", agg)
    json.dump({"summary": summary, "aggregate": agg,
               "runs": [{k: v for k, v in r.items() if k != "genome"} for r in runs]},
              open(os.path.join(HERE, "results", "analysis.json"), "w"), indent=1, default=float)

    # morphology selected by behaviour (A prompts, all seeds)
    A = [r for r in runs if r["prompt_name"].startswith("A_")]
    if A:
        names = sorted({r["prompt_name"] for r in A})
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
        for ax, (key, lab) in zip(axes, [("n_legs", "số chân"), ("mean_leg_len", "chiều dài chân TB (m)"),
                                         ("stand_z", "chiều cao thân khi đứng (m)"), ("torso_W", "bề rộng thân (m)")]):
            for i, n in enumerate(names):
                v = [r["d"][key] for r in A if r["prompt_name"] == n]
                ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.1, 0.1, len(v)), v, s=40)
            ax.set_xticks(range(len(names)), [n[2:] for n in names], rotation=30, fontsize=8)
            ax.set_title(lab, fontsize=10)
        fig.suptitle("Câu lệnh chỉ nói hành vi, thân tự do: tiến hóa chọn thân nào? (3 seed mỗi câu)", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "fig_A_morphology.png"), dpi=140)
        plt.close(fig)

    # gallery: best body per run (seed 0), ours vs T2R for B
    order = sorted({r["prompt_name"] for r in runs})
    cells = []
    for pn in order:
        for m in ("ours", "T2R"):
            rs = [r for r in runs if r["prompt_name"] == pn and r["method"] == m and r["seed"] == 0]
            if not rs:
                continue
            r = rs[0]
            g = json.load(open(r["genome"]))
            img = render.still(g)
            d = r["d"]
            tag = "ours" if m == "ours" else "T2R-style"
            mark = lambda ok: "OK" if ok else "X"
            cells.append(render._label(img, [f"{pn} [{tag}]", f"body {mark(r['body_ok'])}  behaviour {mark(r['behaviour_ok'])}",
                                             f"dx {d['dx']:+.1f} dy {d['dy']:+.1f} yaw {d['yaw']:+.0f} air {d['airtime']:.2f}"]))
            render.render_gif(g, os.path.join(HERE, "gifs", f"{r['run']}.gif"),
                              [f"\"{r['prompt']}\"  [{tag}]",
                               f"{d['n_legs']} legs, leg {d['mean_leg_len']:.2f} m | dx {d['dx']:+.2f} dy {d['dy']:+.2f} "
                               f"yaw {np.degrees(d['yaw']):+.0f} deg air {100 * d['airtime']:.0f}%"])
    if cells:
        cols = 4
        h, w = cells[0].shape[:2]
        rows = (len(cells) + cols - 1) // cols
        canvas = Image.new("RGB", (w * cols, h * rows), "white")
        for i, c in enumerate(cells):
            canvas.paste(Image.fromarray(c), ((i % cols) * w, (i // cols) * h))
        canvas.save(os.path.join(FIGS, "fig_gallery.png"))


if __name__ == "__main__":
    main()
