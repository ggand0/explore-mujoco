"""Test IK controller with random target movements."""
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from controllers.ik_controller import IKController


def test_random_targets():
    """Move end-effector to random targets in workspace."""
    scene_path = Path(__file__).parent / "SO-ARM100/Simulation/SO101/lift_cube_scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    ik = IKController(model, data)

    # Workspace bounds (same as env)
    workspace_min = np.array([0.1, -0.3, 0.01])
    workspace_max = np.array([0.5, 0.3, 0.4])

    # Start with a target above the cube
    target = np.array([0.40, -0.10, 0.15])
    target_hold_steps = 0
    target_threshold = 0.02  # 2cm accuracy

    print(f"Initial EE position: {ik.get_ee_position()}")
    print(f"Target: {target}")
    print("\nControls:")
    print("  - IK will move arm to random targets")
    print("  - New target generated when reached or after timeout")
    print("  - Close viewer to exit")

    step_count = 0
    gripper_action = 1.0  # Start open

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Compute and apply control
            ctrl = ik.step_toward_target(target, gripper_action=gripper_action, gain=0.5)
            data.ctrl[:] = ctrl

            # Step simulation
            mujoco.mj_step(model, data)
            step_count += 1

            # Check distance to target
            current = ik.get_ee_position()
            error = np.linalg.norm(target - current)

            # If reached target or timeout, generate new target
            if error < target_threshold:
                target_hold_steps += 1
                if target_hold_steps > 50:  # Hold for 50 steps
                    # Generate new random target
                    target = np.random.uniform(workspace_min, workspace_max)
                    target_hold_steps = 0
                    gripper_action = np.random.choice([-1.0, 1.0])  # Random gripper
                    print(f"New target: {target}, gripper: {'close' if gripper_action < 0 else 'open'}")
            elif step_count % 500 == 0:
                # Timeout - new target every 500 steps if stuck
                target = np.random.uniform(workspace_min, workspace_max)
                target_hold_steps = 0
                print(f"Timeout - new target: {target}, error was: {error:.3f}")

            # Update viewer at ~30fps
            if step_count % 16 == 0:
                viewer.sync()
                time.sleep(0.01)


def test_cube_approach():
    """Test approaching and grasping the cube."""
    scene_path = Path(__file__).parent / "SO-ARM100/Simulation/SO101/lift_cube_scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # Need to step forward to get sensor data
    mujoco.mj_forward(model, data)

    ik = IKController(model, data)

    # Get cube position (cube is 2cm box at z=0.01, so top is at z=0.02)
    cube_pos = data.sensor("cube_pos").data.copy()
    print(f"Cube position: {cube_pos}")
    print(f"Initial EE position: {ik.get_ee_position()}")

    # The gripperframe site is at the fingertip area
    # Need to descend low enough that fingers are around the cube
    # Cube center at z=0.01, cube is 2cm (±0.01), so top at 0.02
    # Gripper needs to go to roughly cube center height to pinch it
    grasp_height = cube_pos[2]  # Go to cube center height

    phases = [
        {"target": np.array([cube_pos[0], cube_pos[1], 0.10]), "gripper": 1.0, "name": "approach above"},
        {"target": np.array([cube_pos[0], cube_pos[1], grasp_height]), "gripper": 1.0, "name": "descend to cube"},
        {"target": np.array([cube_pos[0], cube_pos[1], grasp_height]), "gripper": -1.0, "name": "close gripper"},
        {"target": np.array([cube_pos[0], cube_pos[1], 0.12]), "gripper": -1.0, "name": "lift"},
    ]

    for i, p in enumerate(phases):
        print(f"  Phase {i} ({p['name']}): target={p['target']}")

    phase_idx = 0
    phase_steps = 0
    phase_complete_steps = 0

    print(f"\nStarting Phase 0: {phases[0]['name']}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            phase = phases[phase_idx]
            target = phase["target"]
            gripper_action = phase["gripper"]

            # Compute and apply control
            ctrl = ik.step_toward_target(target, gripper_action=gripper_action, gain=0.5)
            data.ctrl[:] = ctrl

            # Step simulation
            mujoco.mj_step(model, data)
            phase_steps += 1

            # Check if phase complete
            current = ik.get_ee_position()
            error = np.linalg.norm(target - current)

            # Debug output every 200 steps
            if phase_steps % 200 == 0:
                print(f"  Step {phase_steps}: EE={current}, target={target}, error={error:.4f}")

            if error < 0.02:  # 2cm threshold
                phase_complete_steps += 1
                if phase_complete_steps > 50:  # Hold for 50 steps
                    if phase_idx < len(phases) - 1:
                        phase_idx += 1
                        phase_steps = 0
                        phase_complete_steps = 0
                        print(f"\n=== Phase {phase_idx}: {phases[phase_idx]['name']} ===")
                        new_cube_pos = data.sensor("cube_pos").data.copy()
                        print(f"  Cube now at: {new_cube_pos}")
            else:
                phase_complete_steps = 0

            # Print status during close gripper and lift phases
            if phase_idx >= 2 and phase_steps % 50 == 0:
                cube_z = data.sensor("cube_pos").data[2]
                # Get gripper state
                gripper_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
                gripper_qpos_addr = model.jnt_qposadr[gripper_joint_id]
                gripper_state = data.qpos[gripper_qpos_addr]
                print(f"  cube_z={cube_z:.3f}, gripper={gripper_state:.3f}, ncon={data.ncon}")

            # Update viewer
            if phase_steps % 16 == 0:
                viewer.sync()
                time.sleep(0.01)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "grasp":
        test_cube_approach()
    else:
        test_random_targets()
