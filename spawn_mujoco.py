# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Spawn one Text2Robot URDF in MuJoCo: no text, no training.

The URDF is loaded unchanged except for a <mujoco> compiler block (mesh
directory). A free joint makes the base floating, a floor is added, and each
revolute joint gets a position servo holding 0 rad (the URDF has no actuators;
the real robot's servos hold their zero pose when powered). The robot is
dropped from just above the ground and recorded for a few seconds.

Usage: python spawn_mujoco.py [urdf_path] [out.gif]
"""

import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "Text2Robot", "legged_env", "assets", "urdf", "example_frog", "frog.urdf")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "spawn", "frog_mujoco.gif")
SECONDS = 4.0
KP = 3.0                # servo stiffness, Nm/rad (URDF effort limit is 3.8 Nm)


def build(urdf_path):
    text = open(urdf_path).read()
    compiler = ('<mujoco><compiler meshdir="meshes" strippath="true" balanceinertia="true" '
                'discardvisual="false" fusestatic="false"/></mujoco>')
    head, rest = text.split("<robot", 1)
    name_end = rest.index(">") + 1
    text = head + "<robot" + rest[:name_end] + compiler + rest[name_end:]

    spec = mujoco.MjSpec.from_string(text)
    spec.meshdir = os.path.join(os.path.dirname(os.path.abspath(urdf_path)), "meshes")

    base = spec.worldbody.first_body()
    base.add_freejoint()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
                            rgba=[0.85, 0.85, 0.82, 1])
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            a = spec.add_actuator(name="servo_" + j.name, target=j.name,
                                  trntype=mujoco.mjtTrn.mjTRN_JOINT)
            # position servo: force = kp * (ctrl - q) - kv * qdot
            a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            a.gainprm[0] = KP
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.biasprm[1], a.biasprm[2] = -KP, -0.05
            a.forcerange = [-3.8, 3.8]
            a.forcelimited = 1
            a.ctrlrange = [-1.2, 1.2]
            a.ctrllimited = 1
    spec.option.timestep = 0.002
    return spec.compile()


def main():
    model = build(URDF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # put the lowest point of the robot 2 cm above the floor
    zmin = np.inf
    for g in range(model.ngeom):
        m = model.geom_dataid[g]
        if model.geom_bodyid[g] == 0 or m < 0:
            continue
        v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        world = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        zmin = min(zmin, world[:, 2].min())
    data.qpos[2] += 0.02 - zmin
    mujoco.mj_forward(model, data)

    print(f"bodies {model.nbody - 1}, hinge joints {model.nu}, total mass {model.body_subtreemass[1]:.3f} kg")
    renderer = mujoco.Renderer(model, 360, 480)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 0.9, 135, -20
    frames, z = [], []
    steps_per_frame = int(1 / 30 / model.opt.timestep)
    for i in range(int(SECONDS / model.opt.timestep)):
        mujoco.mj_step(model, data)
        z.append(data.qpos[2])
        if i % steps_per_frame == 0:
            cam.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, cam)
            frames.append(renderer.render())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    imageio.mimsave(OUT, frames, duration=1000 / 30, loop=0)
    imageio.imwrite(OUT.replace(".gif", ".png"), frames[-1])
    print(f"base height: start {z[0]:.3f} m, end {z[-1]:.3f} m; joint angles end "
          f"{np.round(data.qpos[7:], 3).tolist()}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
