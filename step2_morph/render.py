# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Render morph genomes: GIF of a rollout, or a still gallery of bodies."""

import os

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import morph

FPS = 30
try:
    FONT = ImageFont.truetype("arial.ttf", 15)
except OSError:
    FONT = ImageFont.load_default()


def _label(img, lines):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for k, line in enumerate(lines):
        d.text((8, 6 + 18 * k), line, fill=(20, 20, 20), font=FONT)
    return np.asarray(im)


def render_gif(genome, out_path, title_lines=(), seconds=morph.SECONDS, size=(480, 360), dist=1.3):
    frames, start, cache = [], {}, {}

    def cb(model, data, i):
        if "r" not in cache:
            cache["r"] = mujoco.Renderer(model, size[1], size[0])
            cache["cam"] = mujoco.MjvCamera()
            cache["cam"].distance, cache["cam"].azimuth, cache["cam"].elevation = dist, 135, -22
        if "xy" not in start:
            start["xy"] = data.qpos[:2].copy()
        r, cam = cache["r"], cache["cam"]
        cam.lookat[:] = data.qpos[:3]
        r.update_scene(data, cam)
        scn = r.scene
        if scn.ngeom < scn.maxgeom:
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.04, 0.04, 0.001]),
                                np.array([*start["xy"], 0.001]), np.eye(3).flatten(),
                                np.array([0.9, 0.2, 0.2, 1], np.float32))
            scn.ngeom += 1
        frames.append(_label(r.render(), list(title_lines) + [f"t = {i * morph.DT:.1f} s"]))

    res = morph.rollout(genome, seconds=seconds, frame_cb=cb, frame_every=int(1 / FPS / morph.DT))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(out_path, frames, duration=1000 / FPS, loop=0)
    return res


def still(genome, settle=True, size=(320, 260), dist=1.2, az=135, el=-20):
    """Image of the body after the 0.5 s settle with servos holding 0 rad."""
    g = np.array(genome, float).copy()
    g[morph.N_BODY + 1:] = np.tile([0, 0.5, 0.5], morph.N_SLOTS)
    img = {}

    def cb(model, data, i):
        if i == int(0.48 / morph.DT):
            r = mujoco.Renderer(model, size[1], size[0])
            cam = mujoco.MjvCamera()
            cam.distance, cam.azimuth, cam.elevation = dist, az, el
            cam.lookat[:] = data.qpos[:3]
            r.update_scene(data, cam)
            img["x"] = r.render()

    morph.rollout(g, seconds=0.5, frame_cb=cb, frame_every=1)
    return img["x"]
