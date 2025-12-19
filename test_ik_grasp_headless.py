"""Test IK-based grasp with orientation control (headless)."""
import mujoco
import numpy as np
from pathlib import Path
from controllers.ik_controller import IKController

scene_path = Path("SO-ARM100/Simulation/SO101/lift_cube_scene.xml")
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
cube_qpos_addr = model.jnt_qposadr[cube_joint_id]
cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

cube_x, cube_y = 0.40, -0.10

# Place cube
data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = [cube_x, cube_y, 0.015]
data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7] = [1, 0, 0, 0]

# Pre-rotate wrist_roll by 90 degrees so fingers are horizontal
data.qpos[4] = np.pi / 2
data.ctrl[4] = np.pi / 2
mujoco.mj_forward(model, data)

ik = IKController(model, data)

def euler_to_quat(roll, pitch, yaw):
    """Convert euler angles (in radians) to quaternion (w, x, y, z)."""
    cr, cp, cy = np.cos(roll/2), np.cos(pitch/2), np.cos(yaw/2)
    sr, sp, sy = np.sin(roll/2), np.sin(pitch/2), np.sin(yaw/2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])

# Target orientation: gripper pointing down with fingers horizontal
target_quat = euler_to_quat(np.pi/2, np.pi/2, 0)

def get_contacts():
    contacts = []
    for i in range(data.ncon):
        g1, g2 = data.contact[i].geom1, data.contact[i].geom2
        if g1 == cube_geom_id or g2 == cube_geom_id:
            other = g2 if g1 == cube_geom_id else g1
            contacts.append(other)
    return contacts

print("=== IK Grasp Test with Orientation Control (Headless) ===")
print(f"Cube at: ({cube_x}, {cube_y}, 0.015)")
print(f"Target quaternion: {target_quat}")

# Step 1: Approach
print("\n--- Step 1: Approach ---")
cube_z = 0.015
approach_pos = np.array([cube_x - 0.05, cube_y, cube_z])
print(f"Approach target: {approach_pos}")

for step in range(300):
    ctrl = ik.step_toward_target(
        approach_pos,
        gripper_action=1.0,
        gain=0.5,
        target_quat=target_quat,
        orientation_weight=0.5
    )
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)

    if step % 100 == 0:
        tcp = ik.get_ee_position()
        f28 = data.geom_xpos[28]
        f30 = data.geom_xpos[30]
        finger_mid = (f28 + f30) / 2
        print(f"  step {step}: tcp Z={tcp[2]:.4f}, finger_mid Z={finger_mid[2]:.4f}")

# Step 2: Forward
print("\n--- Step 2: Forward ---")
grasp_pos = np.array([cube_x, cube_y, cube_z])

for step in range(300):
    ctrl = ik.step_toward_target(
        grasp_pos,
        gripper_action=1.0,
        gain=0.5,
        target_quat=target_quat,
        orientation_weight=0.5
    )
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)

tcp = ik.get_ee_position()
f28 = data.geom_xpos[28]
f30 = data.geom_xpos[30]
finger_mid = (f28 + f30) / 2
cube_pos = data.qpos[cube_qpos_addr:cube_qpos_addr+3]
print(f"TCP: {tcp}")
print(f"Finger mid: {finger_mid}")
print(f"Cube: {cube_pos}")
print(f"Finger mid Z vs Cube Z: {finger_mid[2]:.4f} vs {cube_pos[2]:.4f}")

# Step 3: Close
print("\n--- Step 3: Close ---")
gripper_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
gripper_qpos_addr = model.jnt_qposadr[gripper_joint_id]

for step in range(300):
    ctrl = ik.step_toward_target(
        grasp_pos,
        gripper_action=-1.0,
        gain=0.5,
        target_quat=target_quat,
        orientation_weight=0.5
    )
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)

    if step % 100 == 0:
        gripper_pos = data.qpos[gripper_qpos_addr]
        print(f"  step {step}: gripper={gripper_pos:.3f}, contacts={get_contacts()}")

# Step 4: Lift
print("\n--- Step 4: Lift ---")
lift_pos = np.array([cube_x, cube_y, 0.10])
for step in range(100):
    ctrl = ik.step_toward_target(
        lift_pos,
        gripper_action=-1.0,
        gain=0.5,
        target_quat=target_quat,
        orientation_weight=0.5
    )
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)

contacts = get_contacts()
cube_z_final = data.qpos[cube_qpos_addr+2]
print(f"\n=== Result ===")
print(f"Cube Z: {cube_z_final:.4f}")
print(f"Contacts: {contacts}")
has_27_28 = 27 in contacts or 28 in contacts
has_29_30 = 29 in contacts or 30 in contacts
print(f"Grasping (both fingers): {has_27_28 and has_29_30}")
print(f"Lifted: {cube_z_final > 0.05}")
