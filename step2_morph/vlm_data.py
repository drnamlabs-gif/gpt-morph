# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Self-labelled data for a behaviour-describing VLM (same recipe as the BETSE / xenobot namers).

Genomes: the four evolved demo robots mutated at several strengths (body and/or gait), plus random genomes.
Each rollout -> one composite image (information drawn into the image, not the prompt):
  top-left  : top view of the torso path in the robot's starting frame (forward = up), heading arrows,
              grey = settle + warm-up, colour = scored window
  top-right : torso height and tilt over time, shaded where every foot is off the ground, 34 deg fall line
  bottom    : three camera frames (t = 2, 4, 6 s)
Caption = rule on the measured descriptors (the same quantities the fitness uses), never written by hand.

Usage: python vlm_data.py <n> [out_dir]       e.g. python vlm_data.py 20 vlm/preview
"""

import io
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = ["B_six_forward_s0", "B_four_right_s0", "L_turn_it_s0", "L_hop_fr_s0"]
SIGMAS = [0.03, 0.08, 0.15, 0.3]
SIZE = 448
CAP = int(os.environ.get("VLM_CAP", "300"))    # max images per behaviour class


def sample_genome(k, rng):
    import morph
    if k % 5 == 4:
        return rng.random(morph.N_GENES), "random"
    seed = SEEDS[k % len(SEEDS)]
    g = np.array(json.load(open(os.path.join(HERE, "results", seed, "best_genome.json"))))
    s = SIGMAS[rng.integers(len(SIGMAS))]
    part = ["all", "body", "gait"][rng.integers(3)]
    mask = np.zeros(morph.N_GENES, bool)
    mask[{"all": slice(None), "body": slice(0, morph.N_BODY), "gait": slice(morph.N_BODY, None)}[part]] = True
    g = np.clip(g + mask * rng.normal(0, s, morph.N_GENES), 0, 1)
    return g, f"{seed}|{part}|{s}"


def caption(d):
    """Behaviour sentence from the measured descriptors (scored window, 4.5 s)."""
    dx, dy, yaw, air = d["dx"], d["dy"], math.degrees(d["yaw"]), d["airtime"]
    travel = math.hypot(dx, dy)
    if d["fallen_at"] is not None:
        return "falls over"
    parts = []
    if air > 0.15:
        parts.append("hops")
    if travel >= 0.4:
        ang = math.degrees(math.atan2(dy, dx))
        dirn = ("forward" if abs(ang) <= 45 else "backward" if abs(ang) >= 135 else
                "sideways to its left" if ang > 0 else "sideways to its right")
        verb = "moves" if parts else "walks"
        if abs(yaw) > 90:
            parts.append(f"{verb} {dirn} while turning {'left' if yaw > 0 else 'right'}")
        else:
            parts.append(f"{verb} {dirn}")
    elif abs(yaw) > 270:
        parts.append(f"spins {'left' if yaw > 0 else 'right'} in place")
    elif travel >= 0.15 or abs(yaw) > 90:
        parts.append("drifts slowly without a clear direction")
    elif not parts:
        parts.append("moves its legs but stays in place" if d["peak_joint_speed"] > 1.0 else "stands still")
    return " and ".join(parts)


def render_one(args):
    k, seed = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco
    from PIL import Image

    import morph
    rng = np.random.default_rng(seed * 100003 + k)
    g, src = sample_genome(k, rng)
    body = morph.decode_body(g)
    log, frames, r = [], [], {}
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 1.1 + 0.9 * max(sum(p["len"]) for p in body["pairs"]), 180, -22
    keep = {int(t / morph.DT) for t in (2.0, 4.0, 5.98)}

    def cb(m, dd, i):
        if "feet" not in r:
            r["feet"] = [j for j in range(m.ngeom) if m.geom_type[j] == mujoco.mjtGeom.mjGEOM_SPHERE]
            r["floor"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if i % 10 == 0:
            touch = any(r["floor"] in (c.geom1, c.geom2) for c in dd.contact[:dd.ncon])
            lift = min(dd.geom_xpos[j][2] for j in r["feet"]) > morph.FOOT_R + morph.LIFT
            log.append([i * morph.DT, *dd.qpos[:7], (not touch) and lift])
        if i in keep:
            if "r" not in r:
                r["r"] = mujoco.Renderer(m, 160, 200)
            cam.lookat[:] = dd.qpos[:3]
            r["r"].update_scene(dd, cam)
            frames.append(r["r"].render())

    d = morph.rollout(g, frame_cb=cb, frame_every=1)
    a = np.array(log, float)
    t, x, y, z, q, air = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4:8], a[:, 8] > 0.5
    w, qx, qy, qz = q.T
    yaw = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (qx * qx + qy * qy), -1, 1)))
    s0 = np.searchsorted(t, 0.5)
    x, y = x - x[s0], y - y[s0]
    scored = t >= 1.5

    sig = "|".join(f"{p['n_seg']}{''.join(p['axis'])}" for p in body["pairs"])
    keys = ("dx", "dy", "yaw", "airtime", "fallen_at", "peak_joint_speed", "n_legs", "n_joints")
    out = []
    for mirror in (False, True):
        sgn = -1 if mirror else 1
        dm = dict(d, dy=sgn * d["dy"], yaw=sgn * d["yaw"])
        img = draw(t, x, sgn * y, z, sgn * yaw, tilt, air, s0, scored, [f[:, ::-1] if mirror else f for f in frames])
        out.append((img, {"id": f"{k:05d}{'m' if mirror else 'o'}", "source": src + ("|mirror" if mirror else ""),
                          "body_sig": sig, "caption": caption(dm), **{kk: dm[kk] for kk in keys}}))
    return out


def draw(t, x, y, z, yaw, tilt, air, s0, scored, frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    fig = plt.figure(figsize=(SIZE / 100, SIZE / 100), dpi=100)
    a1 = fig.add_axes([0.08, 0.42, 0.42, 0.52])
    a1.plot(-y[s0:][~scored[s0:]], x[s0:][~scored[s0:]], color="#aaaaaa", lw=1.5)
    a1.plot(-y[scored], x[scored], color="#d1495b", lw=2.2)
    a1.plot(0, 0, "o", color="k", ms=4)
    for j in range(s0, len(t), 50):
        a1.arrow(-y[j], x[j], -0.12 * math.sin(yaw[j]), 0.12 * math.cos(yaw[j]), head_width=0.05,
                 color="#2b59c3", lw=1)
    lim = max(0.5, np.abs(np.r_[x[s0:], y[s0:]]).max() * 1.15)
    a1.set_xlim(-lim, lim)
    a1.set_ylim(-lim, lim)
    a1.set_aspect("equal")
    a1.set_xticks([])
    a1.set_yticks([])
    a1.set_title("top view (forward = up)", fontsize=7)
    a1.text(0.02, 0.02, f"{2 * lim:.1f} m across", transform=a1.transAxes, fontsize=6)
    a2 = fig.add_axes([0.58, 0.68, 0.38, 0.26])
    a2.fill_between(t, 0, 1, where=air, transform=a2.get_xaxis_transform(), color="#8ecae6", alpha=0.6, lw=0)
    a2.plot(t, z, color="k", lw=1)
    a2.set_title("height (blue = all feet off ground)", fontsize=7)
    a2.tick_params(labelsize=5)
    a3 = fig.add_axes([0.58, 0.42, 0.38, 0.18])
    a3.plot(t, tilt, color="#6a4c93", lw=1)
    a3.axhline(34, color="r", lw=0.8, ls="--")
    a3.set_ylim(0, max(60, tilt.max() * 1.05))
    a3.set_title("tilt deg (red = fall line)", fontsize=7)
    a3.tick_params(labelsize=5)
    for j, fr in enumerate(frames[:3]):
        ai = fig.add_axes([0.02 + j * 0.33, 0.02, 0.31, 0.33])
        ai.imshow(fr)
        ai.set_xticks([])
        ai.set_yticks([])
        ai.set_title(f"t = {[2, 4, 6][j]} s", fontsize=6, pad=1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return Image.open(buf).convert("RGB")


def slug(cap):
    return cap.replace(" ", "_").replace("/", "_")


def main(n, out):
    """out/images/<behaviour>/<id>.png, out/labels.csv (Excel), out/labels.jsonl, out/contact_sheet.png"""
    import csv
    from collections import Counter
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    rows, thumbs = [], []
    with ProcessPoolExecutor(8) as pool:
        for pair in pool.map(render_one, [(k, 0) for k in range(n)], chunksize=4):
          for img, meta in pair:
            if sum(r["caption"] == meta["caption"] for r in rows) >= CAP:
                continue
            folder = os.path.join(out, "images", slug(meta["caption"]))
            os.makedirs(folder, exist_ok=True)
            meta["file"] = os.path.join("images", slug(meta["caption"]), meta["id"] + ".png")
            img.save(os.path.join(out, meta["file"]))
            rows.append(meta)
            if len(thumbs) < 48:
                thumbs.append((img, meta["caption"]))
            if len(rows) % 100 == 0:
                print(len(rows), flush=True)
    with open(os.path.join(out, "labels.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cols = ["id", "caption", "file", "source", "body_sig", "n_legs", "n_joints", "dx", "dy", "yaw", "airtime",
            "fallen_at", "peak_joint_speed"]
    with open(os.path.join(out, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    t = 300
    sheet = Image.new("RGB", (6 * t, ((len(thumbs) + 5) // 6) * (t + 22)), "white")
    d = ImageDraw.Draw(sheet)
    for j, (img, cap) in enumerate(thumbs):
        x, y = (j % 6) * t, (j // 6) * (t + 22)
        sheet.paste(img.resize((t, t)), (x, y + 22))
        d.text((x + 4, y + 4), cap, fill=(200, 30, 30), font=ImageFont.truetype("arial.ttf", 13))
    sheet.save(os.path.join(out, "contact_sheet.png"))
    for cap, m in Counter(r["caption"] for r in rows).most_common():
        print(f"{m:5d}  {cap}")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "vlm", "data"))
