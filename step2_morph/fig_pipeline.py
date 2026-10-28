# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Figure: language -> spec -> co-evolved body + gait -> verification, with the two verified runs.

Usage: python fig_pipeline.py      -> figs/pipeline.png, figs/pipeline.pdf
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import analyze
import morph
import render

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = ["B_six_forward_s0", "B_four_right_s0"]
C_LANG, C_BODY, C_CPG, C_SIM, C_VER = "#5b8def", "#e8a33d", "#3daa77", "#888888", "#c94f4f"
AXIS_WORD = {"x": "sideways swing", "y": "fore-aft swing", "z": "yaw"}


def load(run):
    r = os.path.join(HERE, "results", run)
    return json.load(open(os.path.join(r, "summary.json"))), json.load(open(os.path.join(r, "best_genome.json")))


def track(genome):
    """Torso xy every 10 ms, and four fixed-camera frames along the run."""
    body = morph.decode_body(genome)
    xy, frames, cache = [], [], {}
    full = morph.rollout(genome, frame_cb=lambda m, d, i: xy.append(d.qpos[:2].copy()), frame_every=5)
    xy = np.array(xy)
    centre = np.r_[(xy.min(0) + xy.max(0)) / 2, 0.1]
    span = np.ptp(xy, axis=0).max()
    keep = {int(t / morph.DT) for t in np.linspace(0.5, 5.98, 7)}

    def cb(m, d, i):
        if i not in keep:
            return
        if "r" not in cache:
            cache["r"] = mujoco.Renderer(m, 340, 640)
            cam = mujoco.MjvCamera()
            cam.distance, cam.azimuth, cam.elevation = 0.9 + 0.75 * span, 200, -48
            cam.lookat[:] = centre
            cache["cam"] = cam
        cache["r"].update_scene(d, cache["cam"])
        frames.append(cache["r"].render())

    morph.rollout(genome, frame_cb=cb, frame_every=1)
    return body, xy, frames, full


def box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=color + "22", ec=color, lw=1.6))
    ax.text(x + w / 2, y + h - 0.04, title, ha="center", va="top", fontsize=11, weight="bold", color=color)
    ax.text(x + w / 2, y + h - 0.17, "\n".join(lines), ha="center", va="top", fontsize=9, linespacing=1.35)


def arrow(ax, a, b, color="#444"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=color))


def panel_flow(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    W, H, Y = 0.172, 0.86, 0.02
    xs = [0.012, 0.21, 0.408, 0.606, 0.804]
    box(ax, xs[0], Y, W, H, "1  Prompt", ['"a four-legged robot', 'that walks sideways', 'to the right"', "",
                                          "free English text"], C_LANG)
    box(ax, xs[1], Y, W, H, "2  LLM compiler", ["local LLM, JSON schema", "only allowed fields & ops", "",
                                                "maximise  -10 * dy", "subject to  n_legs == 4", "",
                                                "invalid -> errors sent back", "-> repair"], C_LANG)
    box(ax, xs[2], Y, W, H, "3  One genome, 79 genes", ["24 body genes", "torso size, # leg pairs,",
                                                        "segments, lengths, joint axes", "",
                                                        "55 gait genes (CPG)", "q*(t) = b + A sin(2 pi f t + phi)",
                                                        "per joint: A, phi, b; shared f"], C_BODY)
    box(ax, xs[3], Y, W, H, "4  Evolve body + gait", ["genome -> URDF -> MuJoCo, 6 s", "servo: 3.8 N m, 5.4 rad/s",
                                                     "", "fitness = objective", "  - body-constraint penalty",
                                                     "  - fall penalty (tilt > 34 deg)", "worst of friction {1.0, 0.7}",
                                                     "GA 64 x 150 generations"], C_SIM)
    box(ax, xs[4], Y, W, H, "5  Verify", ["human-written pass rules,", "fixed before the run,",
                                          "independent of the LLM", "", "held-out physics:",
                                          "friction 0.85, dt 1 ms", "", "-> GIF + printable URDF"], C_VER)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + W + 0.003, Y + H / 2), (b - 0.003, Y + H / 2))
    ax.text(0.5, 0.985, "No reinforcement learning: body and gait are found together by one evolutionary search",
            ha="center", va="top", fontsize=10, style="italic", color="#333")


def panel_genome(ax, genome, body):
    g = np.asarray(genome)
    ax.set_xlim(0, morph.N_GENES)
    ax.set_ylim(-1.6, 1.4)
    ax.axis("off")
    cmap_b, cmap_c = plt.get_cmap("Oranges"), plt.get_cmap("Greens")
    for i, v in enumerate(g):
        ax.add_patch(Rectangle((i, 0), 0.92, 1, fc=(cmap_b if i < morph.N_BODY else cmap_c)(0.2 + 0.7 * v), ec="none"))
    groups = [(0, 2, "torso\nL, W"), (2, 3, "#\npairs")]
    for p in range(morph.MAX_PAIRS):
        o = 3 + p * 7
        groups.append((o, o + 7, f"pair {p + 1}: #seg,\n3 lengths, 3 axes"))
    groups += [(morph.N_BODY, morph.N_BODY + 1, "f")]
    groups += [(morph.N_BODY + 1, morph.N_GENES, "18 joint slots x (A, phi, b)  -- unused slots ignored")]
    for a, b, lab in groups:
        ax.plot([a, b - 0.08], [-0.15, -0.15], color="#333", lw=1)
        ax.text((a + b) / 2, -0.3, lab, ha="center", va="top", fontsize=7.5)
    ax.text(morph.N_BODY / 2, 1.15, "body genes (24)", ha="center", fontsize=9.5, weight="bold", color=C_BODY)
    ax.text((morph.N_BODY + morph.N_GENES) / 2, 1.15, "gait genes (55)", ha="center", fontsize=9.5, weight="bold",
            color=C_CPG)
    ax.text(0, -1.45, "Evolved genome of the right-walking robot (colour = gene value in [0, 1]). "
            "Discrete genes (# pairs, # segments, axis) are read by binning [0, 1].", fontsize=8, color="#444")


def panel_frames(ax, frames, title):
    ax.imshow(np.min(np.stack(frames), axis=0))       # time-lapse: robot is darker than the floor
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9.5, loc="left")


def panel_track(ax, xy, d):
    t = np.arange(len(xy)) * 5 * morph.DT
    s = int(0.5 / (5 * morph.DT))
    xy = xy - xy[s]
    w = t >= 1.5
    ax.plot(xy[~w, 1] * -1, xy[~w, 0], color="#bbb", lw=1.4, label="settle + warm-up (not scored)")
    ax.plot(xy[w, 1] * -1, xy[w, 0], color=C_SIM, lw=2.0, label="scored window, 4.5 s")
    ax.scatter([0], [0], color="#c33", s=30, zorder=5, label="start")
    ax.scatter([-xy[-1, 1]], [xy[-1, 0]], color="k", marker="s", s=24, zorder=5, label="end")
    lim = max(0.6, np.abs(xy).max() * 1.15)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="#ddd", lw=0.8, zorder=0)
    ax.axvline(0, color="#ddd", lw=0.8, zorder=0)
    ax.set_xlabel("robot's right  (m)", fontsize=8.5)
    ax.set_ylabel("robot's forward  (m)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_title("top view of torso path", fontsize=9.5, loc="left")
    ax.legend(fontsize=6.8, loc="lower left" if xy[-1, 0] > 0 else "upper left", frameon=False)


def panel_card(ax, s, d, body, checks, label):
    spec = s["spec"]
    ax.axis("off")
    beh = "  ".join(f"{t['weight']:+g}*{t['descriptor']}" for t in spec["behaviour"] if t["weight"])
    con = ", ".join(f"{c['descriptor']} {c['op']} {c['value']:g}" for c in spec["morphology"])
    axes = [AXIS_WORD[a] for p in body["pairs"] for a in p["axis"]]
    ax_count = {w: axes.count(w) for w in AXIS_WORD.values() if axes.count(w)}
    ok = lambda v: "PASS" if v else "FAIL"
    rows = [(label, 12, "bold", "#111"),
            (f'"{spec["prompt"]}"', 10, "normal", C_LANG),
            ("", 4, "normal", "k"),
            ("LLM spec", 9, "bold", C_LANG),
            (f"maximise {beh}", 9, "normal", "#222"),
            (f"subject to {con}", 9, "normal", "#222"),
            ("", 4, "normal", "k"),
            ("Evolved body", 9, "bold", C_BODY),
            (f"{d['n_legs']} legs, {d['n_joints']} servos, segments/pair {[p['n_seg'] for p in body['pairs']]}", 9,
             "normal", "#222"),
            ("joint axes: " + ", ".join(f"{n}x {w}" for w, n in ax_count.items()), 9, "normal", "#222"),
            (f"torso {body['L']:.2f} x {body['W']:.2f} m, {d['mass']:.2f} kg, stands {d['stand_z']:.2f} m", 9,
             "normal", "#222"),
            ("", 4, "normal", "k"),
            ("Behaviour (scored 4.5 s)", 9, "bold", C_SIM),
            (f"forward {d['dx']:+.2f} m, right {-d['dy']:+.2f} m, turn {np.degrees(d['yaw']):+.0f} deg, no fall",
             9, "normal", "#222"),
            ("", 4, "normal", "k"),
            ("Verification", 9, "bold", C_VER),
            (f"body rule {ok(checks[0])}   behaviour rule {ok(checks[1])}", 9, "normal", "#222"),
            (checks[2], 9, "normal", "#222")]
    h_pt = ax.get_position().height * ax.figure.get_figheight() * 72
    y = 1.0
    for txt, fs, wt, c in rows:
        ax.text(0, y, txt, fontsize=fs, weight=wt, color=c, va="top", transform=ax.transAxes)
        y -= 1.55 * fs / h_pt


def held_out(genome, key):
    out = []
    x = morph.rollout(genome, friction=0.85)
    out.append(x[key])
    morph.DT = 0.001
    try:
        out.append(morph.rollout(genome)[key])
    finally:
        morph.DT = 0.002
    return out


def main():
    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(4, 3, height_ratios=[0.62, 0.42, 1.0, 1.0], width_ratios=[1.0, 1.9, 0.85],
                          hspace=0.28, wspace=0.12, left=0.02, right=0.985, top=0.965, bottom=0.02)
    fig.suptitle("GPT-Morph: from language to a printable robot that moves", fontsize=16, weight="bold")
    panel_flow(fig.add_subplot(gs[0, :]))

    data = []
    for run in RUNS:
        s, g = load(run)
        body, xy, frames, full = track(g)
        data.append((run, s, g, body, xy, frames, full))
    panel_genome(fig.add_subplot(gs[1, :]), data[1][2], data[1][3])

    for row, (run, s, g, body, xy, frames, full) in enumerate(data):
        d = s["descriptors"]
        key, sign, word = ("dx", 1, "forward") if "forward" in run else ("dy", -1, "right")
        ho = held_out(g, key)
        checks = tuple(analyze.check(s["spec"]["name"], d)) + (
            f"held-out: friction 0.85 -> {sign * ho[0]:.2f} m, dt 1 ms -> {sign * ho[1]:.2f} m {word}",)
        panel_card(fig.add_subplot(gs[2 + row, 0]), s, d, body, checks, "AB"[row])
        panel_frames(fig.add_subplot(gs[2 + row, 1]), frames, "time-lapse, fixed camera: 7 moments from t = 0.5 s to 6 s")
        panel_track(fig.add_subplot(gs[2 + row, 2]), xy, d)

    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figs", f"pipeline.{ext}"), dpi=130)
    print("saved figs/pipeline.png")


if __name__ == "__main__":
    main()
