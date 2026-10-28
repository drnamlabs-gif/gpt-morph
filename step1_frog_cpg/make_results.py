# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Step-1 results: GIF per behaviour, frame strips, specificity heatmap, LaTeX table."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import evolve
import figs_robot
import frog
import render

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "doc", "figs")
VI = {"forward": "đi tới (+x)", "backward": "đi lùi (−x)", "sideways": "đi ngang sang trái (+y)",
      "turn_left": "xoay trái tại chỗ", "hop": "nhảy (cả 4 chân rời đất)"}


def strip(gif, out, fracs=(0.12, 0.35, 0.6, 0.95)):
    im = Image.open(gif)
    n = im.n_frames
    frames = []
    for f in fracs:
        im.seek(int((n - 1) * f))
        frames.append(im.convert("RGB"))
    w, h = frames[0].size
    canvas = Image.new("RGB", (w * len(frames), h))
    for i, fr in enumerate(frames):
        canvas.paste(fr, (i * w, 0))
    canvas.save(out)


def main():
    keys = list(evolve.BEHAVIOURS)
    rows = []
    for k in keys:
        s = json.load(open(os.path.join(RES, k, "summary.json")))
        g = np.array(json.load(open(os.path.join(RES, k, "best_genome.json"))))
        gif = os.path.join(HERE, "gifs", f"{k}.gif")
        d = s["descriptors"]
        render.render(g, gif, [f'"{s["prompt"]}"',
                               f"dx {d['dx']:+.2f} m  dy {d['dy']:+.2f} m  yaw {np.degrees(d['yaw']):+.0f} deg  "
                               f"air {100 * d['airtime']:.0f}%"])
        strip(gif, os.path.join(FIGS, f"strip_{k}.png"))
        rows.append((k, s))
        if k == "forward":
            figs_robot.fig_servo_cpg(g, label="champion 'đi tới'")

    spec = json.load(open(os.path.join(RES, "specificity.json")))
    M = np.array([[spec[r]["scores"][c] for c in keys] for r in keys])
    Z = (M - M.mean(0)) / (M.std(0) + 1e-9)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-2, vmax=2)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    weight="bold" if i == j else "normal")
    ax.set_xticks(range(len(keys)), [f"metric\n{k}" for k in keys], fontsize=8)
    ax.set_yticks(range(len(keys)), [f"champion\n{k}" for k in keys], fontsize=8)
    ax.set_title("Mỗi champion chấm trên mọi metric\n(màu = chuẩn hóa theo cột; đậm = metric của chính nó)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_specificity.png"), dpi=150)
    plt.close(fig)
    own_best = [int(np.argmax(M[:, j]) == j) for j in range(len(keys))]

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for k, _ in rows:
        h = json.load(open(os.path.join(RES, k, "history.json")))
        ax.plot([x["best"] for x in h], label=k)
    ax.set_xlabel("thế hệ")
    ax.set_ylabel("fitness tốt nhất")
    ax.legend(fontsize=8, ncol=5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_curves.png"), dpi=150)
    plt.close(fig)

    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Câu lệnh & Hành vi & $\Delta x$ (m) & $\Delta y$ (m) & xoay (°) & bay (\%) & $f$ (Hz) & đúng metric? \\",
             r"\midrule"]
    for (k, s), ok in zip(rows, own_best):
        d = s["descriptors"]
        lines.append(f"\\texttt{{{s['prompt'] if k != 'hop' else 'hop (all feet off ground)'}}} & {VI[k]} & {d['dx']:+.2f} & {d['dy']:+.2f} & "
                     f"{np.degrees(d['yaw']):+.0f} & {100 * d['airtime']:.0f} & {s['cpg']['f_hz']:.2f} & "
                     f"{'có' if ok else 'không'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(HERE, "doc", "results_table.tex"), "w", encoding="utf-8").write("\n".join(lines))
    s0 = rows[0][1]
    meta = {"evals_per_behaviour": s0["evaluations"], "steps_per_eval": s0["sim_steps_per_eval"],
            "wall": {k: round(s["wall_clock_s"]) for k, s in rows}, "own_best": dict(zip(keys, own_best)),
            "peak_speed": {k: s["descriptors"]["peak_joint_speed"] for k, s in rows},
            "overspeed": {k: s["descriptors"]["overspeed_frac"] for k, s in rows}}
    json.dump(meta, open(os.path.join(RES, "meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
