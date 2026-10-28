# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Render a Frog CPG genome to GIF (follow camera, heading arrow on the floor)."""

import json
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import frog

FPS = 30


def render(genome, out_path, title_lines=(), seconds=frog.SECONDS, size=(480, 360)):
    if frog._MODEL is None:
        frog._MODEL = frog.build_model()
    model = frog._MODEL
    renderer = mujoco.Renderer(model, size[1], size[0])
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 1.1, 135, -25
    frames = []
    start = {}

    def cb(data, i):
        if "xy" not in start:
            start["xy"] = data.qpos[:2].copy()
        cam.lookat[:] = data.qpos[:3]
        renderer.update_scene(data, cam)
        # start marker and +x heading arrow (visual only)
        scn = renderer.scene
        if scn.ngeom < scn.maxgeom - 2:
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.04, 0.04, 0.001]),
                                np.array([*start["xy"], 0.001]), np.eye(3).flatten(),
                                np.array([0.9, 0.2, 0.2, 1], np.float32))
            scn.ngeom += 1
        img = Image.fromarray(renderer.render())
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except OSError:
            font = ImageFont.load_default()
        for k, line in enumerate(list(title_lines) + [f"t = {i * frog.DT:.1f} s"]):
            d.text((8, 6 + 18 * k), line, fill=(20, 20, 20), font=font)
        frames.append(np.asarray(img))

    res = frog.rollout(genome, seconds=seconds, frame_cb=cb, frame_every=int(1 / FPS / frog.DT))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(out_path, frames, duration=1000 / FPS, loop=0)
    return res


if __name__ == "__main__":
    g = np.array(json.load(open(sys.argv[1]))) if sys.argv[1].endswith(".json") else \
        np.full(frog.N_GENES, float(sys.argv[1]))
    print(render(g, sys.argv[2], [os.path.basename(sys.argv[2])]))
