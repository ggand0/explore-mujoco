"""Test IK-based grasp - targeting finger position, not TCP."""
import mujoco
import numpy as np
import imageio
from pathlib import Path
from controllers.ik_controller import IKController

scene_path = Path("SO-ARM100/Simulation/SO101/lift_cube_scene.xml")
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

renderer = mujoco.Renderer(model, height=480, width=640)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.30, 0.0, 0.05]
cam.distance = 0.5
cam.azimuth = 135
cam.elevation = -25

cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
cube_qpos_addr = model.jnt_qposadr[cube_joint_id]
cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

# Cube position - close enough for arm to reach
cube_x, cube_y = 0.32, 0.0

# Place cube
data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = [cube_x, cube_y, 0.015]
data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7] = [1, 0, 0, 0]

# Pre-rotate wrist_roll by 90 degrees so fingers are horizontal
data.qpos[4] = np.pi / 2
data.ctrl[4] = np.pi / 2
mujoco.mj_forward(model, data)

ik = IKController(model, data)

def get_finger_mid():
    f28 = data.geom_xpos[28]
    f30 = data.geom_xpos[30]
    return (f28 + f30) / 2

def get_tcp_to_finger_offset():
    """Get offset from TCP to finger midpoint in world coords."""
    tcp = ik.get_ee_position()
    finger_mid = get_finger_mid()
    return tcp - finger_mid

def get_contacts():
    contacts = []
    for i in range(data.ncon):
        g1, g2 = data.contact[i].geom1, data.contact[i].geom2
        if g1 == cube_geom_id or g2 == cube_geom_id:
            other = g2 if g1 == cube_geom_id else g1
            contacts.append(other)
    return contacts

def capture_frame():
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()

frames = []
output_dir = Path("runs/ik_grasp_test")
output_dir.mkdir(parents=True, exist_ok=True)

print("=== IK Grasp Test - Finger Position Targeting ===")
print(f"Cube at: ({cube_x}, {cube_y}, 0.015)")

frames.append(capture_frame())

# Target orientation from good grasp pose
target_quat = np.array([0.56531494, -0.5384444, 0.45249218, 0.43098432])

# Step 1: Approach - target finger_mid at (cube_x, cube_y, cube_z)
print("\n--- Step 1: Approach ---")
cube_z = 0.015

for step in range(400):
    # Calculate where TCP needs to be so fingers are at cube position
    offset = get_tcp_to_finger_offset()
    # Target: fingers at (cube_x - 0.03, cube_y, cube_z) - slightly before cube
    finger_target = np.array([cube_x - 0.03, cube_y, cube_z])
    tcp_target = finger_target + offset

    # Lock wrist joints - IK controls only base, shoulder, elbow
    ctrl = ik.step_toward_target(tcp_target, gripper_action=1.0, gain=0.5, locked_joints=[3, 4])
    ctrl[3] = 1.65  # Force wrist_flex to point down
    ctrl[4] = np.pi / 2  # Keep wrist roll for horizontal fingers
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)

    if step % 10 == 0:
        frames.append(capture_frame())

    if step % 100 == 0:
        tcp = ik.get_ee_position()
        finger_mid = get_finger_mid()
        print(f"  step {step}: finger_mid Z={finger_mid[2]:.4f}, tcp Z={tcp[2]:.4f}")

# Step 2: Forward to cube
print("\n--- Step 2: Forward ---")

for step in range(400):
    offset = get_tcp_to_finger_offset()
    finger_target = np.array([cube_x, cube_y, cube_z])
    tcp_target = finger_target + offset

    ctrl = ik.step_toward_target(tcp_target, gripper_action=1.0, gain=0.5, locked_joints=[3, 4])
    ctrl[3] = 1.65
    ctrl[4] = np.pi / 2
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)

    if step % 10 == 0:
        frames.append(capture_frame())

tcp = ik.get_ee_position()
finger_mid = get_finger_mid()
cube_pos = data.qpos[cube_qpos_addr:cube_qpos_addr+3]
print(f"TCP: {tcp}")
print(f"Finger mid: {finger_mid}")
print(f"Cube: {cube_pos}")
print(f"Finger mid Z vs Cube Z: {finger_mid[2]:.4f} vs {cube_pos[2]:.4f}")

# Step 3: Close
print("\n--- Step 3: Close ---")
gripper_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
gripper_qpos_addr = model.jnt_qposadr[gripper_joint_id]

for step in range(800):
    t = min(step / 600, 1.0)  # Slower gripper close
    gripper_action = 1.0 - 2.0 * t

    offset = get_tcp_to_finger_offset()
    finger_target = np.array([cube_x, cube_y, cube_z])
    tcp_target = finger_target + offset

    ctrl = ik.step_toward_target(tcp_target, gripper_action=gripper_action, gain=0.5, locked_joints=[3, 4])
    ctrl[3] = 1.65
    ctrl[4] = np.pi / 2
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)

    if step % 10 == 0:
        frames.append(capture_frame())

    if step % 100 == 0:
        gripper_pos = data.qpos[gripper_qpos_addr]
        print(f"  step {step}: gripper={gripper_pos:.3f}, contacts={get_contacts()}")

# Step 4: Lift
print("\n--- Step 4: Lift ---")
for step in range(150):
    offset = get_tcp_to_finger_offset()
    finger_target = np.array([cube_x, cube_y, 0.10])
    tcp_target = finger_target + offset

    ctrl = ik.step_toward_target(tcp_target, gripper_action=-1.0, gain=0.5, locked_joints=[3, 4])
    ctrl[3] = 1.65
    ctrl[4] = np.pi / 2
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)

    if step % 5 == 0:
        frames.append(capture_frame())

contacts = get_contacts()
cube_z_final = data.qpos[cube_qpos_addr+2]
print(f"\n=== Result ===")
print(f"Cube Z: {cube_z_final:.4f}")
print(f"Contacts: {contacts}")
has_27_28 = 27 in contacts or 28 in contacts
has_29_30 = 29 in contacts or 30 in contacts
print(f"Grasping (both fingers): {has_27_28 and has_29_30}")
print(f"Lifted: {cube_z_final > 0.05}")

# Save video
video_path = output_dir / "ik_grasp_finger_target.mp4"
imageio.mimsave(video_path, frames, fps=30)
print(f"\nSaved {len(frames)} frames to {video_path}")

renderer.close()
