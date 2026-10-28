# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Verify one run: body, behaviour, robustness on held-out physics, intent check, GIF.

Usage: python verify.py <run_name>      e.g. B_four_right_s0
"""

import json
import os
import sys

import numpy as np
from PIL import Image

import analyze
import morph
import render

HERE = os.path.dirname(os.path.abspath(__file__))


def main(run):
    r = os.path.join(HERE, "results", run)
    s = json.load(open(os.path.join(r, "summary.json")))
    g = json.load(open(os.path.join(r, "best_genome.json")))
    d, b, spec = s["descriptors"], s["body"], s["spec"]
    morph.configure(s["spec"].get("physics"))
    pname = spec["name"].replace("__T2R", "")
    print(f'prompt: "{spec["prompt"]}"')
    print("spec:   maximise", " + ".join(f"{t['weight']:g}*{t['descriptor']}" for t in spec["behaviour"]),
          "| body", [f"{c['descriptor']} {c['op']} {c['value']}" for c in spec.get("morphology", [])])
    print(f"body:   {d['n_legs']} legs, {d['n_joints']} servos, segs/pair {[p['n_seg'] for p in b['pairs']]}, "
          f"axes {[''.join(p['axis']) for p in b['pairs']]}, torso {b['L']:.2f}x{b['W']:.2f} m, "
          f"leg {d['mean_leg_len']:.2f} m, mass {d['mass']:.2f} kg, stand {d['stand_z']:.2f} m")
    print(f"motion: dx {d['dx']:+.2f} dy {d['dy']:+.2f} (2nd half dx {d['dx_2nd_half']:+.2f} dy {d['dy_2nd_half']:+.2f}) "
          f"yaw {np.degrees(d['yaw']):+.0f} deg air {d['airtime']:.2f} fallen {d['fallen_at']} peak {d['peak_joint_speed']:.1f} rad/s")
    for mu in (0.7, 0.85):
        x = morph.rollout(g, friction=mu)
        print(f"  friction {mu}: dx {x['dx']:+.2f} dy {x['dy']:+.2f} yaw {np.degrees(x['yaw']):+.0f} fallen {x['fallen_at']}")
    morph.DT = 0.001
    x = morph.rollout(g)
    morph.DT = 0.002
    print(f"  dt 1 ms:      dx {x['dx']:+.2f} dy {x['dy']:+.2f} yaw {np.degrees(x['yaw']):+.0f} fallen {x['fallen_at']}")
    tilts = []
    morph.rollout(g, frame_cb=lambda m, dd, i: tilts.append(np.degrees(np.arccos(np.clip(morph._upz(dd.qpos[3:7]), -1, 1))))
                  if i % 50 == 0 else None, frame_every=1)
    print(f"max torso tilt {max(tilts):.0f} deg")
    print("intent check (body, behaviour):", analyze.check(pname, d))
    gif = os.path.join(HERE, "gifs", f"{run}.gif")
    render.render_gif(g, gif, [f'"{spec["prompt"]}"', f"{d['n_legs']} legs, {d['n_joints']} servos | "
                                                   f"dx {d['dx']:+.2f} dy {d['dy']:+.2f} m in 4.5 s"])
    im = Image.open(gif)
    frames = []
    for f in (0.1, 0.4, 0.7, 0.98):
        im.seek(int((im.n_frames - 1) * f))
        frames.append(im.convert("RGB"))
    w, h = frames[0].size
    out = Image.new("RGB", (w * 4, h))
    for i, fr in enumerate(frames):
        out.paste(fr, (i * w, 0))
    out.save(os.path.join(HERE, "figs", f"strip_{run}.png"))


if __name__ == "__main__":
    main(sys.argv[1])
