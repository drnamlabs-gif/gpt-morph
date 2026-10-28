# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Text2Robot Frog in MuJoCo + open-loop CPG + behaviour metrics.

Body: the unchanged Text2Robot example Frog URDF (1 body, 4 legs x 2 links,
8 revolute joints). Servo gains copied from Text2Robot's own task config
(RobotDog.yaml): PD stiffness 3.8 Nm/rad, damping 0.1 Nm s/rad, torque limit
3.8 Nm, joint range +-1.2 rad, ground friction 1.0.

Servo speed: the URDF gives velocity 5.4 rad/s (HTD-45H no-load speed), which
MuJoCo does not enforce by itself. Each servo is modelled as a DC motor driven
by a PD loop (see servo_torque): stall torque 3.8 N m, no-load speed 5.4 rad/s,
back-EMF always opposing motion.
"""

import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "..", "Text2Robot", "legged_env", "assets", "urdf", "example_frog", "frog.urdf")

KP, KD, TORQUE = 3.8, 0.1, 3.8
OMEGA_MAX = 5.4                      # rad/s, URDF <limit velocity>
Q_LIM = 1.2
DT = 0.002
CTRL_EVERY = 4                       # Text2Robot decimation: control at 125 Hz
SECONDS = 6.0
SOLREF_T = 0.004                     # contact time constant (MuJoCo default 0.02 is soft)
FRICTIONS = (1.0, 0.7)               # robust evaluation: worst case over these
N_JOINTS = 8
BOUNDS = {"f": (0.5, 3.0), "A": (0.0, 0.8), "phi": (0.0, 2 * np.pi), "b": (-0.4, 0.4)}
N_GENES = 1 + 3 * N_JOINTS


def build_model(friction=1.0):
    text = open(URDF).read()
    compiler = ('<mujoco><compiler meshdir="meshes" strippath="true" balanceinertia="true" '
                'discardvisual="false" fusestatic="false"/></mujoco>')
    head, rest = text.split("<robot", 1)
    k = rest.index(">") + 1
    spec = mujoco.MjSpec.from_string(head + "<robot" + rest[:k] + compiler + rest[k:])
    spec.meshdir = os.path.join(os.path.dirname(URDF), "meshes")
    spec.worldbody.first_body().add_freejoint()
    floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1],
                                    rgba=[0.85, 0.85, 0.82, 1], friction=[1.0, 0.005, 0.0001])
    floor.material = ""
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8], castshadow=True)
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            j.damping = KD + TORQUE / OMEGA_MAX          # servo damping + back-EMF
            a = spec.add_actuator(name="servo_" + j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.gaintype = mujoco.mjtGain.mjGAIN_FIXED     # ctrl = torque (N m), computed by servo_torque
            a.gainprm[0] = 1.0
            a.ctrlrange = [-TORQUE, TORQUE]
            a.ctrllimited = 1
    spec.option.timestep = DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    model = spec.compile()
    for g in range(model.ngeom):
        model.geom_friction[g, 0] = friction
        model.geom_solref[g, 0] = SOLREF_T     # stiff contacts: no bouncing on soft contact
    return model


def drop_height(model):
    """Base z so that the lowest mesh vertex starts 1 cm above the floor."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    zmin = np.inf
    for g in range(model.ngeom):
        m = model.geom_dataid[g]
        if model.geom_bodyid[g] == 0 or m < 0:
            continue
        v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        zmin = min(zmin, (v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min())
    return data.qpos[2] + 0.01 - zmin


def decode(genome):
    g = np.clip(np.asarray(genome, float), 0, 1)
    lo, hi = BOUNDS["f"]
    f = lo + g[0] * (hi - lo)
    p = g[1:].reshape(N_JOINTS, 3)
    A = BOUNDS["A"][0] + p[:, 0] * (BOUNDS["A"][1] - BOUNDS["A"][0])
    phi = p[:, 1] * 2 * np.pi
    b = BOUNDS["b"][0] + p[:, 2] * (BOUNDS["b"][1] - BOUNDS["b"][0])
    return f, A, phi, b


def cpg_targets(genome, t):
    f, A, phi, b = decode(genome)
    return np.clip(b + A * np.sin(2 * np.pi * f * t + phi), -Q_LIM, Q_LIM)


def servo_torque(model, data, q_target):
    """Hobby-servo model: PD sets the motor voltage, back-EMF opposes motion.

    u   = clip(KP (q* - q) / TORQUE, -1, 1)          normalised voltage
    tau = TORQUE u  -  (KD + TORQUE / OMEGA_MAX) qdot
    Only TORQUE u is applied here; the velocity term is the joint damping set
    in build_model, which the implicit integrator handles stably (the leg links
    are too light for an explicit damping torque at dt = 2 ms).
    Stall torque = TORQUE; no-load speed = OMEGA_MAX / (1 + KD OMEGA_MAX / TORQUE).
    """
    q = data.qpos[7:7 + N_JOINTS]
    return TORQUE * np.clip(KP * (q_target - q) / TORQUE, -1.0, 1.0)


def yaw_of(quat):
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def up_z(quat):
    w, x, y, z = quat
    return 1 - 2 * (x * x + y * y)


_MODEL = None
_MODELS = {}


def rollout(genome, seconds=SECONDS, frame_cb=None, frame_every=None, friction=1.0):
    """Run one episode. Returns behaviour descriptors (all in SI units)."""
    global _MODEL
    if friction not in _MODELS:
        _MODELS[friction] = build_model(friction)
    model = _MODEL = _MODELS[friction]
    data = mujoco.MjData(model)
    data.qpos[2] = drop_height(model)
    mujoco.mj_forward(model, data)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    x0, y0 = data.qpos[0], data.qpos[1]
    yaw_prev, yaw_total = yaw_of(data.qpos[3:7]), 0.0
    n_steps = int(seconds / DT)
    settle = int(0.5 / DT)                   # first 0.5 s: hold pose, robot lands
    airborne, counted, flipped_at, peak_qd, over, n_qd = 0, 0, None, 0.0, 0, 0
    z_hist = []
    for i in range(n_steps):
        if i % CTRL_EVERY == 0:
            t = max(0, i - settle) * DT
            q_target = np.zeros(N_JOINTS) if i < settle else cpg_targets(genome, t)
        data.ctrl[:] = servo_torque(model, data, q_target)
        mujoco.mj_step(model, data)
        if i >= settle:
            qd_abs = np.abs(data.qvel[6:6 + N_JOINTS])
            peak_qd = max(peak_qd, float(qd_abs.max()))
            over += int((qd_abs > OMEGA_MAX).sum())
            n_qd += N_JOINTS
        if frame_cb is not None and i % frame_every == 0:
            frame_cb(data, i)
        if i < settle:
            if i == settle - 1:
                x0, y0 = data.qpos[0], data.qpos[1]
                yaw_prev, yaw_total = yaw_of(data.qpos[3:7]), 0.0
            continue
        if flipped_at is None and up_z(data.qpos[3:7]) < 0.5:
            flipped_at = i * DT
        if flipped_at is not None:
            continue
        yaw = yaw_of(data.qpos[3:7])
        yaw_total += (yaw - yaw_prev + np.pi) % (2 * np.pi) - np.pi
        yaw_prev = yaw
        touching = any(model.geom_bodyid[c.geom1] > 0 and c.geom2 == floor_id or
                       model.geom_bodyid[c.geom2] > 0 and c.geom1 == floor_id
                       for c in data.contact[:data.ncon])
        airborne += not touching
        counted += 1
        z_hist.append(data.qpos[2])
        x_end, y_end = data.qpos[0], data.qpos[1]
    if counted == 0:
        x_end, y_end = x0, y0
    z_hist = np.array(z_hist) if z_hist else np.array([0.0])
    return {"dx": float(x_end - x0), "dy": float(y_end - y0), "yaw": float(yaw_total),
            "airtime": airborne / max(counted, 1), "z_mean": float(z_hist.mean()),
            "z_range": float(np.percentile(z_hist, 95) - np.percentile(z_hist, 5)),
            "flipped_at": flipped_at, "peak_joint_speed": peak_qd,
            "overspeed_frac": over / max(n_qd, 1)}
