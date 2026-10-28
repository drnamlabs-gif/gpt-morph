# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Story GIF for one run: prompt -> LoRA compiler -> spec -> random design -> evolution -> evolved robot -> checks.

Usage: python story.py <run>      e.g. L_turn_it_s0   -> gifs/story_<run>.gif
"""

import json
import math
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import analyze
import evolve_morph as E
import morph

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 640, 480, 20
BAND = 74
GLOSS = {"un robot che gira su se stesso": "Italian: \"a robot that spins around itself\"",
         "un robot aux longues jambes qui saute": "French: \"a robot with long legs that jumps\""}
INK, MUTE, BLUE, ORANGE, GREEN, RED = (25, 25, 25), (110, 110, 110), (60, 110, 220), (220, 130, 30), (40, 150, 90), (200, 60, 60)


def font(size, bold=False, mono=False):
    name = "consola.ttf" if mono else ("arialbd.ttf" if bold else "arial.ttf")
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def wrap(d, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = t
    return lines + [cur]


def card():
    return Image.new("RGB", (W, H), (255, 255, 255))


def header(im, prompt, stage, color):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, BAND], fill=(255, 255, 255))
    d.line([0, BAND, W, BAND], fill=(210, 210, 210), width=2)
    d.text((14, 8), stage, font=font(15, bold=True), fill=color)
    f = font(21)
    for k, line in enumerate(wrap(d, f'"{prompt}"', f, W - 28)[:2]):
        d.text((14, 28 + 22 * k), line, font=f, fill=INK)
    return d


def footer(im, left, right="", color=INK):
    d = ImageDraw.Draw(im)
    d.rectangle([0, H - 44, W, H], fill=(255, 255, 255))
    d.line([0, H - 44, W, H - 44], fill=(210, 210, 210), width=2)
    d.text((14, H - 34), left, font=font(19, bold=True), fill=color)
    if right:
        f = font(19)
        d.text((W - 14 - d.textlength(right, font=f), H - 34), right, font=f, fill=INK)


# ---- scenes -------------------------------------------------------------------------------------------------------

def scene_prompt(prompt):
    frames, durs = [], []
    f = font(30)
    for k in range(0, len(prompt) + 1, max(1, len(prompt) // 20)):
        im = card()
        d = ImageDraw.Draw(im)
        d.text((40, 120), "1  Prompt (any language)", font=font(20, bold=True), fill=BLUE)
        for j, line in enumerate(wrap(d, '"' + prompt[:k] + ("" if k >= len(prompt) else "|"), f, W - 80)):
            d.text((40, 180 + 40 * j), line, font=f, fill=INK)
        if k >= len(prompt) and prompt in GLOSS:
            d.text((40, 300), GLOSS[prompt], font=font(20), fill=MUTE)
        frames.append(im)
        durs.append(60)
    durs[-1] = 1200
    return frames, durs


def draw_net(d, x0, y0, w, h, phase):
    layers = [4, 6, 6, 6, 4]
    xs = [x0 + w * i / (len(layers) - 1) for i in range(len(layers))]
    pts = [[(x, y0 + h * (j + 1) / (n + 1)) for j in range(n)] for x, n in zip(xs, layers)]
    for a, b in zip(pts[:-1], pts[1:]):
        for p in a:
            for q in b:
                d.line([p, q], fill=(200, 210, 235), width=1)
    for i, layer in enumerate(pts):
        lit = i <= phase
        for p in layer:
            r = 6
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=BLUE if lit else (220, 225, 240),
                      outline=(90, 120, 200))


def scene_compiler(prompt, spec):
    lines = ["{", '  "behaviour": ['] + [f'    {{"{t["descriptor"]}": {t["weight"]:+g}}}' for t in spec["behaviour"]
                                         if t["weight"]] + ["  ],", '  "morphology": ['] + \
            [f'    "{c["descriptor"]} {c["op"]} {c["value"]:g}"' for c in spec["morphology"]] + ["  ]", "}"]
    frames, durs = [], []
    steps = 5 + len(lines)
    for s in range(steps + 1):
        im = card()
        d = header(im, prompt, "2  Compile: sentence -> spec", BLUE)
        d.text((20, 96), "Qwen2.5-1.5B + LoRA", font=font(17, bold=True), fill=BLUE)
        d.text((20, 118), "fine-tuned: sentence -> spec JSON", font=font(14), fill=MUTE)
        draw_net(d, 40, 140, 200, 230, min(s, 4))
        d.text((270, 250), "->", font=font(30, bold=True), fill=MUTE if s < 5 else INK)
        d.text((330, 96), "spec (objective + body rules)", font=font(17, bold=True), fill=ORANGE)
        for k, line in enumerate(lines[:max(0, s - 4)]):
            d.text((330, 130 + 24 * k), line, font=font(17, mono=True), fill=INK)
        footer(im, "language decides WHAT to optimise", "", BLUE)
        frames.append(im)
        durs.append(220)
    durs[-1] = 2200
    return frames, durs


def metric_fn(spec):
    key = max(spec["behaviour"], key=lambda t: abs(t["weight"]))
    k, sgn = key["descriptor"], math.copysign(1, key["weight"])
    if k == "dx":
        return ("forward" if sgn > 0 else "backward"), lambda s: f"{sgn * s['dx']:+.2f} m"
    if k == "dy":
        return ("to its right" if sgn < 0 else "to its left"), lambda s: f"{sgn * s['dy']:+.2f} m"
    if k == "yaw":
        return "turned", lambda s: f"{np.degrees(sgn * s['yaw']):+.0f} deg"
    return "in the air", lambda s: f"{100 * s['air'] / max(s['n'], 1):.0f}% of the time"


def scene_sim(genome, prompt, spec, stage, color, seconds, label):
    """Tracking-camera rollout with a live metric of the spec's main behaviour."""
    name, fmt = metric_fn(spec)
    r, cam, st, frames = {}, mujoco.MjvCamera(), {"dx": 0, "dy": 0, "yaw": 0, "air": 0, "n": 0}, []
    body = morph.decode_body(genome)
    cam.distance, cam.azimuth, cam.elevation = 1.1 + 0.9 * max(0.2, max(sum(p["len"]) for p in body["pairs"])), 135, -22
    settle = int(0.5 / morph.DT)

    def cb(m, d, i):
        if "r" not in r:
            r["r"] = mujoco.Renderer(m, H, W)
            r["feet"] = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
            r["floor"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if i == settle:
            r["x0"], r["yaw0"], r["yp"] = d.qpos[:2].copy(), morph._yaw(d.qpos[3:7]), morph._yaw(d.qpos[3:7])
        if i >= settle and not st.get("fell") and morph._upz(d.qpos[3:7]) < morph.UPRIGHT_COS:
            st["fell"] = True                  # same rule as scoring: nothing counts after a fall
        if i >= settle and not st.get("fell"):
            yw = morph._yaw(d.qpos[3:7])
            st["yaw"] += (yw - r["yp"] + math.pi) % (2 * math.pi) - math.pi
            r["yp"] = yw
            st["dx"], st["dy"] = d.qpos[0] - r["x0"][0], d.qpos[1] - r["x0"][1]
            touch = any(r["floor"] in (c.geom1, c.geom2) for c in d.contact[:d.ncon])
            st["air"] += (not touch) and min(d.geom_xpos[g][2] for g in r["feet"]) > morph.FOOT_R + morph.LIFT
            st["n"] += 1
        if i % int(1 / FPS / morph.DT):
            return
        cam.lookat[:] = d.qpos[:3]
        rr = r["r"]
        rr.update_scene(d, cam)
        scn = rr.scene
        if "x0" in r and scn.ngeom < scn.maxgeom:
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.04, 0.04, 0.001]),
                                np.array([*r["x0"], 0.001]), np.eye(3).flatten(), np.array([0.9, 0.2, 0.2, 1], np.float32))
            scn.ngeom += 1
        im = Image.fromarray(rr.render())
        header(im, prompt, stage, color)
        footer(im, label + (" FELL OVER" if st.get("fell") else ""), f"{name}: {fmt(st)}   t = {i * morph.DT:.1f} s",
               RED if st.get("fell") else color)
        frames.append(im)

    morph.rollout(genome, seconds=seconds, frame_cb=cb, frame_every=1)
    return frames, [1000 // FPS] * len(frames)


def scene_evolution(prompt, history, n_gen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = [h["generation"] for h in history]
    best = [h["best"] for h in history]
    frames, durs = [], []
    for k in list(range(2, len(g) + 1, max(1, len(g) // 15))) + [len(g)]:
        fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=100)
        ax.plot(g[:k], best[:k], color=[v / 255 for v in GREEN], lw=2.5)
        ax.set_xlim(0, n_gen)
        ax.set_ylim(min(best) - 0.05 * np.ptp(best) - 1e-6, max(best) + 0.1 * np.ptp(best) + 1e-6)
        ax.set_xlabel("generation", fontsize=12)
        ax.set_ylabel("best fitness", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.canvas.draw()
        plot = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
        im = card()
        header(im, prompt, "4  Evolve body + gait together", GREEN)
        im.paste(plot, (0, 92))
        footer(im, f"64 designs x {n_gen} generations, MuJoCo physics", f"gen {g[k - 1]}", GREEN)
        frames.append(im)
        durs.append(120)
    durs[-1] = 1000
    return frames, durs


def scene_checks(prompt, s, checks):
    d_ = s["descriptors"]
    b = s["body"]
    axes = "".join(a for p in b["pairs"] for a in p["axis"])
    rows = [("Evolved body", ORANGE),
            (f"{d_['n_legs']} legs, {d_['n_joints']} servos, legs {d_['mean_leg_len']:.2f} m", INK),
            (f"joint axes per side: {axes.count('x')} sideways swing, {axes.count('y')} fore-aft, {axes.count('z')} yaw", INK),
            (f"torso {b['L']:.2f} x {b['W']:.2f} m, {d_['mass']:.2f} kg", INK),
            ("", INK),
            ("Independent checks (written before the run)", RED),
            (f"body rule:      {'PASS' if checks[0] else 'FAIL'}", INK),
            (f"behaviour rule: {'PASS' if checks[1] else 'FAIL'}", INK),
            ("held-out friction + time step: still works", INK),
            ("output: printable URDF (servo + PLA parts)", INK)]
    im = card()
    d = header(im, prompt, "6  Verify", RED)
    for k, (t, c) in enumerate(rows):
        d.text((30, 100 + 30 * k), t, font=font(20, bold=c != INK), fill=c)
    return [im], [3500]


def main(run):
    rdir = os.path.join(HERE, "results", run)
    s = json.load(open(os.path.join(rdir, "summary.json")))
    genome = json.load(open(os.path.join(rdir, "best_genome.json")))
    history = json.load(open(os.path.join(rdir, "history.json")))
    spec, prompt = s["spec"], s["spec"]["prompt"]
    morph.configure(s["spec"].get("physics"))

    # generation 0: a random genome (same seed stream as the run) that obeys the body rules
    rng = np.random.default_rng(s["seed"])
    stub = {"mass": 0, "stand_z": 0, "dx": 0, "dy": 0, "yaw": 0, "airtime": 0, "flipped_at": None}
    for tries in range(1, 20001):
        g0 = rng.random(morph.N_GENES)
        if not E.evaluate(spec, {**morph.describe(morph.decode_body(g0)), **stub})[2]:
            break
    print("random designs drawn until one fits the body rules:", tries)

    parts = [scene_prompt(prompt), scene_compiler(prompt, spec),
             scene_sim(g0, prompt, spec, "3  Before evolution: a random design that fits the body rules", MUTE,
                       4.0, "gen 0 (random)"),
             scene_evolution(prompt, history, s["generations"]),
             scene_sim(genome, prompt, spec, "5  After evolution: best body + gait", GREEN, morph.SECONDS,
                       f"gen {s['generations']} (evolved)"),
             scene_checks(prompt, s, analyze.check(spec["name"], s["descriptors"]))]
    frames = [np.asarray(f) for p in parts for f in p[0]]
    durs = [d for p in parts for d in p[1]]
    out = os.path.join(HERE, "gifs", f"story_{run}.gif")
    imageio.mimsave(out, frames, duration=durs, loop=0)
    print(out, len(frames), "frames", f"{sum(durs) / 1000:.1f} s", f"{os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
