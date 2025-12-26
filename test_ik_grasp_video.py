"""Test IK grasp with top-down approach - 10 trials."""
import mujoco
import numpy as np
from pathlib import Path
from controllers.ik_controller import IKController

scene_path = Path("SO-ARM100/Simulation/SO101/lift_cube_scene.xml")
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
cube_qpos_addr = model.jnt_qposadr[cube_joint_id]
cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
gripper_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
gripper_qpos_addr = model.jnt_qposadr[gripper_joint_id]

def get_contacts():
    contacts = []
    for i in range(data.ncon):
        g1, g2 = data.contact[i].geom1, data.contact[i].geom2
        if g1 == cube_geom_id or g2 == cube_geom_id:
            other = g2 if g1 == cube_geom_id else g1
            contacts.append(other)
    return contacts

def run_trial(seed):
    np.random.seed(seed)
    mujoco.mj_resetData(model, data)

    # Randomize cube position
    cube_x = 0.32 + np.random.uniform(-0.02, 0.02)
    cube_y = 0.0 + np.random.uniform(-0.02, 0.02)
    cube_z = 0.015

    data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = [cube_x, cube_y, cube_z]
    data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7] = [1, 0, 0, 0]
    data.qpos[4] = np.pi / 2
    data.ctrl[4] = np.pi / 2
    mujoco.mj_forward(model, data)

    ik = IKController(model, data)

    # Step 1: Position above cube
    for _ in range(400):
        target = np.array([cube_x, cube_y, cube_z + 0.05])
        ctrl = ik.step_toward_target(target, gripper_action=1.0, gain=0.5, locked_joints=[3, 4])
        ctrl[3] = 1.65
        ctrl[4] = np.pi / 2
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

    # Step 2: Descend to cube
    for _ in range(400):
        target = np.array([cube_x, cube_y, cube_z])
        ctrl = ik.step_toward_target(target, gripper_action=1.0, gain=0.5, locked_joints=[3, 4])
        ctrl[3] = 1.65
        ctrl[4] = np.pi / 2
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

    # Step 3: Close gripper
    grasp_action = 1.0
    grasped = False
    for step in range(800):
        contacts = get_contacts()
        has_static = any(g in contacts for g in [27, 28])
        has_moving = any(g in contacts for g in [29, 30])

        t = min(step / 600, 1.0)
        gripper_action = 1.0 - 2.0 * t

        target = np.array([cube_x, cube_y, cube_z])
        ctrl = ik.step_toward_target(target, gripper_action=gripper_action, gain=0.5, locked_joints=[3, 4])
        ctrl[3] = 1.65
        ctrl[4] = np.pi / 2
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        gripper_pos = data.qpos[gripper_qpos_addr]
        if has_static and has_moving and gripper_pos < 0.25:
            grasp_action = gripper_action
            grasped = True
            break

    # Step 4: Lift
    for _ in range(200):
        target = np.array([cube_x, cube_y, cube_z + 0.08])
        ctrl = ik.step_toward_target(target, gripper_action=grasp_action, gain=0.5, locked_joints=[3, 4])
        ctrl[3] = 1.65
        ctrl[4] = np.pi / 2
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

    cube_final_z = data.qpos[cube_qpos_addr + 2]
    lifted = cube_final_z > 0.05

    return grasped, lifted, cube_final_z

print("=== 10 Trial Test ===\n")
successes = 0
for seed in range(10):
    grasped, lifted, cube_z = run_trial(seed)
    status = "SUCCESS" if lifted else "FAIL"
    print(f"Trial {seed}: grasped={grasped}, lifted={lifted}, cube_z={cube_z:.3f} [{status}]")
    if lifted:
        successes += 1

print(f"\n=== Result: {successes}/10 ({successes*10}%) ===")
