"""Test IK-based grasp with orientation control."""
import mujoco
import mujoco.viewer
import numpy as np
import time
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
data.qpos[4] = np.pi / 2  # wrist_roll
data.ctrl[4] = np.pi / 2
mujoco.mj_forward(model, data)

ik = IKController(model, data)

# Target orientation: gripper pointing down with fingers horizontal
# Rotation: 90° around Y (pitch down) then 90° around Z (roll for horizontal fingers)
# As quaternion (w, x, y, z):
# Pitch -90° around Y: (cos(-45°), 0, sin(-45°), 0) = (0.707, 0, -0.707, 0)
# Then roll 90° around local Z
# Combined: gripper Z pointing down, fingers along world Y
def euler_to_quat(roll, pitch, yaw):
    """Convert euler angles (in radians) to quaternion (w, x, y, z)."""
    cr, cp, cy = np.cos(roll/2), np.cos(pitch/2), np.cos(yaw/2)
    sr, sp, sy = np.sin(roll/2), np.sin(pitch/2), np.sin(yaw/2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])

# Gripper pointing down: pitch = -90° (or +90° depending on convention)
# With fingers horizontal: roll = 90°
target_quat = euler_to_quat(np.pi/2, np.pi/2, 0)  # roll=90°, pitch=90°
print(f"Target quaternion: {target_quat}")

def get_contacts():
    contacts = []
    for i in range(data.ncon):
        g1, g2 = data.contact[i].geom1, data.contact[i].geom2
        if g1 == cube_geom_id or g2 == cube_geom_id:
            other = g2 if g1 == cube_geom_id else g1
            contacts.append(other)
    return contacts

print("=== IK Grasp Test with Orientation Control ===")
print(f"Cube at: ({cube_x}, {cube_y}, 0.015)")

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Step 1: Move to approach position with orientation
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
        viewer.sync()
        time.sleep(0.01)

        if step % 50 == 0:
            tcp = ik.get_ee_position()
            f28 = data.geom_xpos[28]
            f30 = data.geom_xpos[30]
            finger_mid = (f28 + f30) / 2
            print(f"  step {step}: tcp Z={tcp[2]:.4f}, finger_mid Z={finger_mid[2]:.4f}")

    # Step 2: Move forward to cube
    print("\n--- Step 2: Forward ---")
    grasp_pos = np.array([cube_x, cube_y, cube_z])
    print(f"Grasp target: {grasp_pos}")

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
        viewer.sync()
        time.sleep(0.01)

    tcp = ik.get_ee_position()
    f28 = data.geom_xpos[28]
    f30 = data.geom_xpos[30]
    finger_mid = (f28 + f30) / 2
    cube_pos = data.qpos[cube_qpos_addr:cube_qpos_addr+3]
    print(f"TCP: {tcp}")
    print(f"Finger mid: {finger_mid}")
    print(f"Cube: {cube_pos}")
    print(f"Finger mid Z vs Cube Z: {finger_mid[2]:.4f} vs {cube_pos[2]:.4f}")

    # Step 3: Close gripper
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
        viewer.sync()
        time.sleep(0.01)

        if step % 50 == 0:
            gripper_pos = data.qpos[gripper_qpos_addr]
            print(f"  step {step}: gripper={gripper_pos:.3f}, contacts={get_contacts()}")

    print(f"Contacts: {get_contacts()}")
    print(f"Cube: {data.qpos[cube_qpos_addr:cube_qpos_addr+3]}")
    print(f"Gripper: {data.qpos[gripper_qpos_addr]:.3f}")

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
        viewer.sync()
        time.sleep(0.01)

    contacts = get_contacts()
    print(f"Cube Z: {data.qpos[cube_qpos_addr+2]:.4f}")
    print(f"Contacts: {contacts}")

    has_27_28 = 27 in contacts or 28 in contacts
    has_29_30 = 29 in contacts or 30 in contacts
    print(f"Grasping (both fingers): {has_27_28 and has_29_30}")

    # Keep viewer open
    print("\nPress Ctrl+C to close...")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.1)
