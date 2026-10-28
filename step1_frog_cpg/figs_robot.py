# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Figures explaining the Text2Robot Frog's parts (for the Vietnamese PDF).

fig_parts.png      each link in its own colour, 4 views, world axes at the base
fig_joints.png     what each joint type does (hip about y, knee about x)
fig_tree.png       kinematic tree with masses and joint axes
fig_modules.png    the printed electromechanical modules shipped with Text2Robot
fig_servo_cpg.png  servo PD torque law and an example CPG target signal
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import mujoco
import numpy as np

import frog

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "doc", "figs")
os.makedirs(OUT, exist_ok=True)
MODULES = os.path.join(HERE, "..", "Text2Robot", "STL_Files", "Electronics_Modules")

COLORS = {"base_link": (0.75, 0.75, 0.75),
          "body_1_0_1": (0.85, 0.33, 0.10), "body_2_0_1": (0.99, 0.65, 0.35),
          "body_1_1_1": (0.13, 0.47, 0.71), "body_2_1_1": (0.55, 0.75, 0.93),
          "body_1_2_1": (0.17, 0.63, 0.17), "body_2_2_1": (0.60, 0.87, 0.54),
          "body_1_3_1": (0.58, 0.40, 0.74), "body_2_3_1": (0.80, 0.70, 0.90)}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def coloured_model():
    model = frog.build_model()
    for g in range(model.ngeom):
        b = model.body(model.geom_bodyid[g]).name
        if b in COLORS:
            model.geom_rgba[g] = (*COLORS[b], 1.0)
    return model


def axes_arrows(scn, origin, length=0.18):
    for vec, rgba in [((1, 0, 0), (0.9, 0.1, 0.1, 1)), ((0, 1, 0), (0.1, 0.7, 0.1, 1)),
                      ((0, 0, 1), (0.1, 0.2, 0.9, 1))]:
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.006, np.array(origin),
                             np.array(origin) + length * np.array(vec))
        scn.ngeom += 1


def shot(model, data, az, el, dist, lookat, arrows=True, size=(640, 480)):
    r = mujoco.Renderer(model, size[1], size[0])
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    cam.lookat[:] = lookat
    r.update_scene(data, cam)
    if arrows:
        axes_arrows(r.scene, data.qpos[:3] + np.array([0, 0, 0.16]))
    return r.render()


def fig_parts():
    model = coloured_model()
    data = mujoco.MjData(model)
    data.qpos[2] = frog.drop_height(model) - 0.01
    mujoco.mj_forward(model, data)
    look = data.qpos[:3] + np.array([0, 0, -0.05])
    views = [("Iso view", 135, -25, 0.95), ("Top view (from above)", 90, -89, 0.9),
             ("Front view (looking along -x)", 0, -5, 0.9), ("Side view (looking along +y)", 90, -5, 0.9)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.8))
    for ax, (t, az, el, d) in zip(axes.flat, views):
        ax.imshow(shot(model, data, az, el, d, look))
        ax.set_title(t)
        ax.axis("off")
    names = {"base_link": "thân (base_link)", "body_1_0_1": "đùi chân 0", "body_2_0_1": "cẳng chân 0",
             "body_1_1_1": "đùi chân 1", "body_2_1_1": "cẳng chân 1", "body_1_2_1": "đùi chân 2",
             "body_2_2_1": "cẳng chân 2", "body_1_3_1": "đùi chân 3", "body_2_3_1": "cẳng chân 3"}
    handles = [plt.Line2D([], [], marker="s", ls="", ms=11, color=COLORS[k], label=v) for k, v in names.items()]
    handles += [plt.Line2D([], [], color=c, lw=3, label=l) for c, l in
                [((0.9, 0.1, 0.1), "trục x"), ((0.1, 0.7, 0.1), "trục y"), ((0.1, 0.2, 0.9), "trục z")]]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_parts.png"), dpi=130)
    plt.close(fig)


def fig_joints():
    model = coloured_model()
    data = mujoco.MjData(model)
    z = frog.drop_height(model) + 0.05
    names = [model.joint(j).name for j in range(model.njnt)]
    rows = [("joint_1_0", "Khớp hông joint_1_0 (trục y): đùi chân 0 đánh tới / lui", 0, -5),
            ("joint_2_0", "Khớp gối joint_2_0 (trục x): cẳng chân 0 vung ra / vào ngang thân", 0, -5)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    for g in range(model.ngeom):   # fade everything except leg 0
        if model.body(model.geom_bodyid[g]).name not in ("body_1_0_1", "body_2_0_1"):
            model.geom_rgba[g, 3] = 0.25
    for (jn, title, az, el), row in zip(rows, axes):
        az = -90 if jn == "joint_1_0" else 180   # camera on leg 0's side / in front
        for ax, ang in zip(row, (-0.8, 0.0, 0.8)):
            data.qpos[:] = 0
            data.qpos[2], data.qpos[3] = z, 1.0
            data.qpos[7 + names.index(jn) - 1] = ang
            mujoco.mj_forward(model, data)
            ax.imshow(shot(model, data, az, el, 0.6, data.qpos[:3] + np.array([0.0, 0.07, -0.08]), arrows=False))
            ax.set_title(f"{jn} = {ang:+.1f} rad", fontsize=10)
            ax.axis("off")
        row[1].text(0.5, 1.16, title, transform=row[1].transAxes, ha="center", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_joints.png"), dpi=130)
    plt.close(fig)


def fig_tree():
    model = frog.build_model()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1",
                                    fc=color, ec="#333333", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    mass = {model.body(b).name: model.body_mass[b] for b in range(1, model.nbody)}
    box(38, 37, 24, 7, f"base_link (thân)\n{mass['base_link']*1000:.0f} g", COLORS["base_link"])
    for k in range(4):
        x = 4 + k * 24
        u, l = f"body_1_{k}_1", f"body_2_{k}_1"
        ax.annotate("", (x + 9, 27), (50, 37), arrowprops=dict(arrowstyle="-|>", color="#555"))
        ax.text(x + 9, 30.5, f"joint_1_{k}  hông\ntrục y, ±1.2 rad", ha="center", fontsize=7.5, color="#333")
        box(x, 19, 18, 7, f"{u}\nđùi, {mass[u]*1000:.0f} g", COLORS[u])
        ax.annotate("", (x + 9, 9), (x + 9, 19), arrowprops=dict(arrowstyle="-|>", color="#555"))
        ax.text(x + 9.8, 13.6, f"joint_2_{k}  gối\ntrục x, ±1.2 rad", ha="left", fontsize=7.5, color="#333")
        box(x, 1, 18, 7, f"{l}\ncẳng, {mass[l]*1000:.0f} g", COLORS[l])
    total = sum(mass.values())
    ax.text(99, 44, f"Tổng: {total:.2f} kg\n9 khâu (link), 8 khớp (joint) = 8 servo", ha="right", va="top", fontsize=9)
    fig.savefig(os.path.join(OUT, "fig_tree.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_modules():
    files = [("MotorCase5.stl", "Vỏ servo (MotorCase5)"), ("KneeConnector.stl", "Khớp nối gối (KneeConnector)"),
             ("BatteryBox.stl", "Hộp pin + Raspberry Pi (BatteryBox)"), ("MotorHornforMetal.stl", "Bát servo (MotorHorn)")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, (f, title) in zip(axes, files):
        spec = mujoco.MjSpec()
        spec.meshdir = MODULES
        spec.add_mesh(name="m", file=f, scale=[0.001] * 3)
        b = spec.worldbody.add_body()
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="m", rgba=[0.85, 0.55, 0.25, 1])
        spec.worldbody.add_light(pos=[0.3, -0.3, 0.5], dir=[-0.5, 0.5, -1])
        m = spec.compile()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        size = 2 * m.geom_rbound[0]
        ax.imshow(shot(m, d, 135, -30, 2.2 * size, d.geom_xpos[0], arrows=False, size=(400, 400)))
        ax.set_title(f"{title}\n~{size*100:.0f} cm", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_modules.png"), dpi=130)
    plt.close(fig)


def fig_servo_cpg(genome=None, label="ví dụ ngẫu nhiên"):
    fig, (a, s, b) = plt.subplots(1, 3, figsize=(14, 3.8))
    err = np.linspace(-1.5, 1.5, 300)
    a.plot(err, np.clip(frog.KP * err, -frog.TORQUE, frog.TORQUE), color="#B4561A", lw=2)
    for v in (frog.TORQUE, -frog.TORQUE):
        a.axhline(v, ls=":", color="#888")
    a.set_xlabel("sai số góc  q* − q  (rad)")
    a.set_ylabel("mô-men τ (N·m)")
    a.set_title("(a) Servo khi khớp đứng yên (q̇ = 0)\nτ = 3.8·(q* − q), bão hòa ±3.8 N·m", fontsize=10)

    qd = np.linspace(-9, 9, 300)
    damp = frog.KD + frog.TORQUE / frog.OMEGA_MAX
    up = frog.TORQUE - damp * qd
    lo = -frog.TORQUE - damp * qd
    s.fill_between(qd, lo, up, color="#F2C49B", alpha=0.6, label="mô-men servo tạo được")
    s.plot(qd, up, color="#B4561A")
    s.plot(qd, lo, color="#B4561A")
    s.axvline(frog.OMEGA_MAX, ls="--", color="#2E5E8C")
    s.axvline(-frog.OMEGA_MAX, ls="--", color="#2E5E8C")
    s.axhline(0, color="#999", lw=0.8)
    s.set_ylim(-9, 9)
    s.text(frog.OMEGA_MAX + 0.2, 6.5, "5.4 rad/s\n(giới hạn URDF)", fontsize=8, color="#2E5E8C")
    s.set_xlabel("tốc độ khớp q̇ (rad/s)")
    s.set_title("(b) Motor DC: τ = 3.8·u − 0.80·q̇,  |u| ≤ 1\nquay càng nhanh, lực đẩy càng yếu, lực phanh càng mạnh", fontsize=10)
    s.legend(fontsize=8, loc="lower left")

    if genome is None:
        genome = np.random.default_rng(3).random(frog.N_GENES)
    t = np.linspace(0, 2, 400)
    q = np.array([frog.cpg_targets(genome, x) for x in t])
    for k, col in zip(range(4), ["#D9541A", "#2276B5", "#2CA02C", "#9467BD"]):
        b.plot(t, q[:, k], color=col, lw=1.8, label=f"hông chân {k}")
        b.plot(t, q[:, 4 + k], color=col, lw=1, ls="--")
    f = frog.decode(genome)[0]
    b.set_title(f"(c) CPG ({label}), f = {f:.2f} Hz\nnét liền = hông, nét đứt = gối", fontsize=10)
    b.set_xlabel("thời gian (s)")
    b.set_ylabel("góc đích q* (rad)")
    b.legend(fontsize=7.5, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_servo_cpg.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    fig_parts()
    fig_joints()
    fig_tree()
    fig_modules()
    fig_servo_cpg()
    print("written to", OUT)
