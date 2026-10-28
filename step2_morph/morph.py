# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Parametric legged-robot generator (URDF) + MuJoCo rollout with a CPG.

Body plan, built from Text2Robot's printed modules (sizes measured from
Text2Robot/STL_Files/Electronics_Modules):
  torso  box L x W x 0.09 m, must hold BatteryBox (8.2 x 9.1 x 12.7 cm,
         lying flat): L in [0.14, 0.34], W in [0.11, 0.24]
  legs   1-3 mirrored pairs (2, 4 or 6 legs), evenly spaced along the torso,
         hip bracket 3 cm outside the torso side
  leg    1-3 segments, each = one servo (MotorCase5, 3.6 x 4.4 x 4.5 cm) at its
         proximal joint + a printed beam; segment length in [0.05, 0.15] m;
         each joint axis is x (roll), y (pitch) or z (yaw), the three choices
         Text2Robot also allows; range +-1.2 rad
  foot   sphere r = 1.8 cm at the end of the last segment
Left and right legs of a pair share the body genes (bilateral symmetry, as in
Text2Robot) but every joint has its own CPG genes.

Mass model (assumptions, stated): servo 65 g lumped in its segment; printed
beam 25 x 25 mm at 30% effective PLA density; torso = 450 g electronics
(Raspberry Pi 4, 3S LiPo, converter) + 2 mm PLA shell.

Servo: same as step 1 (Text2Robot gains KP 3.8, KD 0.1, 3.8 N m stall,
5.4 rad/s no-load, back-EMF as implicit joint damping).

Genome: 79 numbers in [0, 1]:
  body  3 + 3 x 7 = 24    (L, W, n_pairs; per pair: n_seg, 3 lengths, 3 axes)
  cpg   1 + 18 x 3 = 55   (f; per joint slot A, phi, b; slots of absent joints unused)
"""

import math

import mujoco
import numpy as np

# ---- design space -----------------------------------------------------------
L_RANGE, W_RANGE, TORSO_H = (0.14, 0.34), (0.11, 0.24), 0.09
SEG_RANGE = (0.05, 0.15)
MAX_PAIRS, MAX_SEG = 3, 3
AXES = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
AXIS_ORDER = ["y", "x", "z"]
HIP_OFFSET, BEAM, FOOT_R, SERVO_BLOCK = 0.03, 0.025, 0.018, 0.045
SERVO_MASS, PLA_RHO, PLA_FILL, ELECTRONICS, SHELL_T = 0.065, 1240.0, 0.30, 0.45, 0.002

# ---- control ----------------------------------------------------------------
KP, KD, TORQUE, OMEGA_MAX, Q_LIM = 3.8, 0.1, 3.8, 5.4, 1.2
DT, CTRL_EVERY, SECONDS = 0.002, 4, 6.0
SOLREF_T = 0.004                     # stiff contacts (MuJoCo default 0.02 lets gaits bounce)
FRICTIONS = (1.0, 0.7)               # robust evaluation: worst case over these
CONDITIONS = [{"friction": f} for f in FRICTIONS]   # physics conditions scored (worst case); see configure()
WARMUP = 1.0                         # s of gait before measuring; a one-off lunge earns nothing
LIFT = 0.005                         # m of foot clearance needed to count as airborne
UPRIGHT_COS = math.cos(0.6)          # torso tilt above 0.6 rad (34 deg) = fallen; nothing counts after
F_RANGE, A_MAX, B_MAX = (0.5, 3.0), 0.8, 0.4

N_BODY = 3 + MAX_PAIRS * (1 + 2 * MAX_SEG)
N_SLOTS = MAX_PAIRS * 2 * MAX_SEG                 # joint slots: pair x side x segment
N_CPG = 1 + 3 * N_SLOTS
N_GENES = N_BODY + N_CPG


def configure(physics):
    """Per-spec physics: {'f_max': Hz cap of the CPG, 'conditions': [{'friction', 'solref', 'foot_scale'}, ...]}.
    Must be called the same way when replaying a genome (decoding of f depends on f_max)."""
    global F_RANGE, CONDITIONS
    physics = physics or {}
    F_RANGE = (0.5, physics.get("f_max", 3.0))
    CONDITIONS = physics.get("conditions", [{"friction": f} for f in FRICTIONS])


def _pick(g, n):
    return min(int(g * n), n - 1)


def decode_body(genome):
    g = np.clip(np.asarray(genome, float), 0, 1)
    body = {"L": L_RANGE[0] + g[0] * (L_RANGE[1] - L_RANGE[0]),
            "W": W_RANGE[0] + g[1] * (W_RANGE[1] - W_RANGE[0]),
            "n_pairs": 1 + _pick(g[2], MAX_PAIRS), "pairs": []}
    for p in range(MAX_PAIRS):
        o = 3 + p * (1 + 2 * MAX_SEG)
        n_seg = 1 + _pick(g[o], MAX_SEG)
        lens = [SEG_RANGE[0] + g[o + 1 + j] * (SEG_RANGE[1] - SEG_RANGE[0]) for j in range(MAX_SEG)]
        axes = [AXIS_ORDER[_pick(g[o + 1 + MAX_SEG + j], 3)] for j in range(MAX_SEG)]
        body["pairs"].append({"n_seg": n_seg, "len": lens[:n_seg], "axis": axes[:n_seg]})
    body["pairs"] = body["pairs"][:body["n_pairs"]]
    return body


def joint_slots(body):
    """List of (slot index, joint name) in the order the joints appear in the URDF."""
    out = []
    for p, pair in enumerate(body["pairs"]):
        for s, side in enumerate(("L", "R")):
            for j in range(pair["n_seg"]):
                out.append(((p * 2 + s) * MAX_SEG + j, f"j_{p}{side}{j}"))
    return out


def describe(body):
    legs = 2 * body["n_pairs"]
    leg_len = [sum(pr["len"]) for pr in body["pairs"]]
    axes = [a for pr in body["pairs"] for a in pr["axis"]]
    return {"n_legs": legs, "n_joints": 2 * sum(pr["n_seg"] for pr in body["pairs"]),
            "torso_L": body["L"], "torso_W": body["W"], "mean_leg_len": float(np.mean(leg_len)),
            "max_leg_len": float(max(leg_len)),
            "axis_frac": {a: axes.count(a) / len(axes) for a in AXIS_ORDER}}


def _inertia_box(m, sx, sy, sz):
    return (m * (sy * sy + sz * sz) / 12, m * (sx * sx + sz * sz) / 12, m * (sx * sx + sy * sy) / 12)


def to_urdf(body, name="morph", foot_r=FOOT_R):
    L, W, H = body["L"], body["W"], TORSO_H
    shell = PLA_RHO * SHELL_T * 2 * (L * W + L * H + W * H)
    m_t = ELECTRONICS + shell
    ixx, iyy, izz = _inertia_box(m_t, L, W, H)
    x = [f'<robot name="{name}">',
         f'<link name="torso"><inertial><mass value="{m_t:.4f}"/>'
         f'<inertia ixx="{ixx:.6f}" iyy="{iyy:.6f}" izz="{izz:.6f}" ixy="0" ixz="0" iyz="0"/></inertial>'
         f'<visual><geometry><box size="{L:.4f} {W:.4f} {H:.4f}"/></geometry></visual>'
         f'<collision><geometry><box size="{L:.4f} {W:.4f} {H:.4f}"/></geometry></collision></link>']
    n = body["n_pairs"]
    xs = [0.0] if n == 1 else list(np.linspace(-L / 2 + 0.04, L / 2 - 0.04, n))
    for p, pair in enumerate(body["pairs"]):
        for side, sgn in (("L", 1), ("R", -1)):
            parent = "torso"
            origin = (xs[p], sgn * (W / 2 + HIP_OFFSET), -H / 2 + 0.02)
            for j in range(pair["n_seg"]):
                l = pair["len"][j]
                link = f"seg_{p}{side}{j}"
                m = SERVO_MASS + PLA_RHO * PLA_FILL * BEAM * BEAM * l
                ixx, iyy, izz = _inertia_box(m, BEAM, BEAM, l)
                geo = (f'<visual><origin xyz="0 0 {-l / 2:.4f}"/><geometry><box size="{BEAM} {BEAM} {l:.4f}"/></geometry></visual>'
                       f'<visual><geometry><box size="{SERVO_BLOCK} {SERVO_BLOCK} {SERVO_BLOCK}"/></geometry></visual>'
                       f'<collision><origin xyz="0 0 {-l / 2:.4f}"/><geometry><box size="{BEAM} {BEAM} {l:.4f}"/></geometry></collision>')
                if j == pair["n_seg"] - 1:
                    geo += (f'<visual><origin xyz="0 0 {-l:.4f}"/><geometry><sphere radius="{foot_r:.4f}"/></geometry></visual>'
                            f'<collision><origin xyz="0 0 {-l:.4f}"/><geometry><sphere radius="{foot_r:.4f}"/></geometry></collision>')
                x.append(f'<link name="{link}"><inertial><origin xyz="0 0 {-l / 2:.4f}"/><mass value="{m:.4f}"/>'
                         f'<inertia ixx="{ixx:.7f}" iyy="{iyy:.7f}" izz="{izz:.7f}" ixy="0" ixz="0" iyz="0"/></inertial>{geo}</link>')
                ax = AXES[pair["axis"][j]]
                x.append(f'<joint name="j_{p}{side}{j}" type="revolute"><parent link="{parent}"/><child link="{link}"/>'
                         f'<origin xyz="{origin[0]:.4f} {origin[1]:.4f} {origin[2]:.4f}"/>'
                         f'<axis xyz="{ax[0]} {ax[1]} {ax[2]}"/>'
                         f'<limit effort="{TORQUE}" velocity="{OMEGA_MAX}" lower="{-Q_LIM}" upper="{Q_LIM}"/></joint>')
                parent, origin = link, (0.0, 0.0, -l)
    x.append("</robot>")
    return "\n".join(x)


def build_model(body, friction=1.0, solref=None, foot_scale=1.0):
    spec = mujoco.MjSpec.from_string(to_urdf(body, foot_r=FOOT_R * foot_scale))
    spec.worldbody.first_body().add_freejoint()
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1],
                            rgba=[0.85, 0.85, 0.82, 1], contype=1, conaffinity=2)
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6])
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            j.damping = KD + TORQUE / OMEGA_MAX
            a = spec.add_actuator(name="servo_" + j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            a.gainprm[0] = 1.0
            a.ctrlrange = [-TORQUE, TORQUE]
            a.ctrllimited = 1
    spec.option.timestep = DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    model = spec.compile()
    for g in range(model.ngeom):
        if model.geom_bodyid[g] > 0:          # robot: touches floor, not itself
            model.geom_contype[g], model.geom_conaffinity[g] = 2, 1
        model.geom_friction[g, 0] = friction
        model.geom_solref[g, 0] = SOLREF_T if solref is None else solref
    return model


def drop_height(model, data):
    mujoco.mj_forward(model, data)
    zmin = np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:
            continue
        r = model.geom_rbound[g] if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE else \
            np.abs(data.geom_xmat[g].reshape(3, 3) @ model.geom_size[g]).max()
        zmin = min(zmin, data.geom_xpos[g][2] - r)
    return data.qpos[2] + 0.01 - zmin


def cpg_targets(genome, slots, t):
    c = np.clip(np.asarray(genome, float)[N_BODY:], 0, 1)
    f = F_RANGE[0] + c[0] * (F_RANGE[1] - F_RANGE[0])
    idx = np.array([s for s, _ in slots])
    p = c[1:].reshape(N_SLOTS, 3)[idx]
    A, phi, b = p[:, 0] * A_MAX, p[:, 1] * 2 * np.pi, (p[:, 2] * 2 - 1) * B_MAX
    return np.clip(b + A * np.sin(2 * np.pi * f * t + phi), -Q_LIM, Q_LIM)


def _yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _upz(q):
    w, x, y, z = q
    return 1 - 2 * (x * x + y * y)


def rollout(genome, seconds=SECONDS, frame_cb=None, frame_every=None, friction=1.0, solref=None, foot_scale=1.0):
    body = decode_body(genome)
    slots = joint_slots(body)
    model = build_model(body, friction, solref, foot_scale)
    data = mujoco.MjData(model)
    data.qpos[2] = drop_height(model, data)
    mujoco.mj_forward(model, data)
    nj = len(slots)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    torso_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == 1}
    feet = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    settle = int(0.5 / DT)
    start = settle + int(WARMUP / DT)     # behaviour is measured from here (steady gait, not the first lunge)
    n = int(seconds / DT)
    q_t = np.zeros(nj)
    x0 = y0 = yaw_prev = yaw_tot = 0.0
    x_half = y_half = None
    air = belly = counted = 0
    flipped, peak = None, 0.0
    z_hist = []
    for i in range(n):
        if i % CTRL_EVERY == 0:
            q_t = np.zeros(nj) if i < settle else cpg_targets(genome, slots, (i - settle) * DT)
        q = data.qpos[7:7 + nj]
        data.ctrl[:] = TORQUE * np.clip(KP * (q_t - q) / TORQUE, -1, 1)
        mujoco.mj_step(model, data)
        if frame_cb is not None and i % frame_every == 0:
            frame_cb(model, data, i)
        if i == settle - 1:
            x0, y0, yaw_prev = data.qpos[0], data.qpos[1], _yaw(data.qpos[3:7])
            stand_z = float(data.qpos[2])
        if i < settle:
            continue
        if flipped is None and _upz(data.qpos[3:7]) < UPRIGHT_COS:   # fallen: tilt > 0.6 rad
            flipped = (i - settle) * DT
        if flipped is not None:
            continue
        if i < start:                     # warm-up: a falling fall still counts, motion does not
            if i == start - 1:
                x0, y0, yaw_prev, yaw_tot = data.qpos[0], data.qpos[1], _yaw(data.qpos[3:7]), 0.0
            continue
        yw = _yaw(data.qpos[3:7])
        yaw_tot += (yw - yaw_prev + math.pi) % (2 * math.pi) - math.pi
        yaw_prev = yw
        touch = torso_touch = False
        for c in data.contact[:data.ncon]:
            if c.geom1 == floor or c.geom2 == floor:
                touch = True
                other = c.geom2 if c.geom1 == floor else c.geom1
                torso_touch |= other in torso_geoms
        # airborne = no floor contact AND every foot clearly lifted (contact flicker of a buzzing foot is not a hop)
        air += (not touch) and min(data.geom_xpos[g][2] for g in feet) > FOOT_R * foot_scale + LIFT
        belly += torso_touch
        counted += 1
        peak = max(peak, float(np.abs(data.qvel[6:6 + nj]).max()))
        z_hist.append(data.qpos[2])
        xe, ye = data.qpos[0], data.qpos[1]
        if i - start == (n - start) // 2:
            x_half, y_half = xe, ye
    if counted == 0:
        xe, ye = x0, y0
    return {"dx": float(xe - x0), "dy": float(ye - y0), "yaw": float(yaw_tot),
            # fractions of the whole scored window: a robot that falls early cannot score 100% airtime
            "airtime": air / max(n - start, 1), "belly_contact": belly / max(n - start, 1),
            "stand_z": stand_z, "z_mean": float(np.mean(z_hist)) if z_hist else 0.0,
            "fallen_at": flipped, "flipped_at": flipped, "peak_joint_speed": peak,
            "dx_2nd_half": float(xe - x_half) if x_half is not None else 0.0,
            "dy_2nd_half": float(ye - y_half) if y_half is not None else 0.0, "mass": float(model.body_subtreemass[1]),
            **describe(body)}
