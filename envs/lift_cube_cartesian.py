"""Lift cube environment with Cartesian (end-effector) action space.

Uses IK controller internally - agent outputs delta XYZ + gripper,
which is much easier to explore than joint velocities.
"""
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from controllers.ik_controller import IKController


class LiftCubeCartesianEnv(gym.Env):
    """Lift cube with Cartesian action space.

    Action space (4 dims):
        - Delta X, Y, Z for end-effector position
        - Gripper open/close (-1 to 1)

    Observation space (21 dims):
        - Joint positions (6)
        - Joint velocities (6)
        - Gripper position (3)
        - Gripper orientation (3) - euler angles
        - Cube position (3)

    This is much easier to learn than joint-space control because:
    - Random actions naturally explore 3D space
    - Agent doesn't need to learn arm kinematics
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        action_scale: float = 0.02,  # 2cm per step max
        lift_height: float = 0.08,
        hold_steps: int = 10,
        reward_type: str = "dense",
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.lift_height = lift_height
        self.hold_steps = hold_steps
        self.reward_type = reward_type
        self._step_count = 0
        self._hold_count = 0

        # Load model
        scene_path = Path(__file__).parent.parent / "SO-ARM100/Simulation/SO101/lift_cube_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        # Initialize IK controller
        self.ik = IKController(self.model, self.data)

        # Joint info
        self.n_joints = 6
        self.ctrl_ranges = self.model.actuator_ctrlrange.copy()

        # Action space: delta XYZ + gripper
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation space
        obs_dim = 6 + 6 + 3 + 3 + 3  # joints pos/vel + gripper pos + gripper euler + cube pos
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Track target EE position
        self._target_ee_pos = None

        # Renderer
        self._renderer = None
        if render_mode == "human":
            self._renderer = mujoco.Renderer(self.model)

    def _get_obs(self) -> np.ndarray:
        joint_pos = self.data.qpos[: self.n_joints].copy()
        joint_vel = self.data.qvel[: self.n_joints].copy()
        gripper_pos = self.ik.get_ee_position()

        # Get gripper orientation as euler angles
        gripper_mat = self.ik.get_ee_orientation()
        gripper_euler = self._rotation_matrix_to_euler(gripper_mat)

        cube_pos = self.data.sensor("cube_pos").data.copy()

        return np.concatenate([
            joint_pos, joint_vel, gripper_pos, gripper_euler, cube_pos
        ]).astype(np.float32)

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to euler angles (roll, pitch, yaw)."""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0

        return np.array([roll, pitch, yaw])

    def _get_gripper_state(self) -> float:
        gripper_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper"
        )
        gripper_qpos_addr = self.model.jnt_qposadr[gripper_joint_id]
        return self.data.qpos[gripper_qpos_addr]

    def _check_cube_contacts(self) -> tuple[bool, bool]:
        """Check if cube contacts static gripper and moving jaw separately."""
        cube_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"
        )
        # Geoms 25-28: static gripper body
        # Geoms 29-30: moving jaw
        static_gripper_geoms = set(range(25, 29))
        moving_jaw_geoms = set(range(29, 31))

        has_gripper_contact = False
        has_jaw_contact = False

        for i in range(self.data.ncon):
            geom1 = self.data.contact[i].geom1
            geom2 = self.data.contact[i].geom2

            other_geom = None
            if geom1 == cube_geom_id:
                other_geom = geom2
            elif geom2 == cube_geom_id:
                other_geom = geom1

            if other_geom is not None:
                if other_geom in static_gripper_geoms:
                    has_gripper_contact = True
                if other_geom in moving_jaw_geoms:
                    has_jaw_contact = True

        return has_gripper_contact, has_jaw_contact

    def _is_grasping(self) -> bool:
        """Check if cube is properly grasped (pinched between gripper and jaw)."""
        gripper_state = self._get_gripper_state()
        is_closed = gripper_state < 0.25

        has_gripper_contact, has_jaw_contact = self._check_cube_contacts()
        is_pinched = has_gripper_contact and has_jaw_contact

        return is_closed and is_pinched

    def _get_info(self) -> dict[str, Any]:
        gripper_pos = self.ik.get_ee_position()
        cube_pos = self.data.sensor("cube_pos").data.copy()

        gripper_to_cube = np.linalg.norm(gripper_pos - cube_pos)
        cube_z = cube_pos[2]
        is_grasping = self._is_grasping()
        is_lifted = is_grasping and cube_z > self.lift_height

        has_gripper_contact, has_jaw_contact = self._check_cube_contacts()
        return {
            "gripper_to_cube": gripper_to_cube,
            "cube_pos": cube_pos.copy(),
            "cube_z": cube_z,
            "gripper_pos": gripper_pos.copy(),
            "gripper_state": self._get_gripper_state(),
            "has_gripper_contact": has_gripper_contact,
            "has_jaw_contact": has_jaw_contact,
            "is_grasping": is_grasping,
            "is_lifted": is_lifted,
            "hold_count": self._hold_count,
            "is_success": self._hold_count >= self.hold_steps,
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Randomize cube position
        if self.np_random is not None:
            cube_x = 0.40 + self.np_random.uniform(-0.03, 0.03)
            cube_y = -0.10 + self.np_random.uniform(-0.03, 0.03)
            cube_joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
            )
            cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
            self.data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = [cube_x, cube_y, 0.01]
            self.data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)

        # Initialize target EE position to current position
        self._target_ee_pos = self.ik.get_ee_position().copy()

        self._step_count = 0
        self._hold_count = 0

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Parse action
        action = np.clip(action, -1.0, 1.0)
        delta_xyz = action[:3] * self.action_scale
        gripper_action = action[3]

        # Update target end-effector position
        self._target_ee_pos += delta_xyz

        # Clamp to workspace bounds
        self._target_ee_pos[0] = np.clip(self._target_ee_pos[0], 0.1, 0.5)
        self._target_ee_pos[1] = np.clip(self._target_ee_pos[1], -0.3, 0.3)
        self._target_ee_pos[2] = np.clip(self._target_ee_pos[2], 0.01, 0.4)

        # Use IK to compute joint controls
        ctrl = self.ik.step_toward_target(
            self._target_ee_pos,
            gripper_action=gripper_action,
            gain=0.5,
        )

        # Apply control and step simulation multiple times for stability
        self.data.ctrl[:] = ctrl
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Get state
        obs = self._get_obs()
        info = self._get_info()

        # Update hold counter
        if info["is_lifted"]:
            self._hold_count += 1
        else:
            self._hold_count = 0
        info["hold_count"] = self._hold_count

        # Check success
        is_success = self._hold_count >= self.hold_steps
        info["is_success"] = is_success

        # Compute reward
        reward = self._compute_reward(info)

        terminated = is_success
        truncated = self._step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, info: dict[str, Any]) -> float:
        if self.reward_type == "sparse":
            return 0.0 if info["is_success"] else -1.0
        else:
            # Dense reward (robosuite-style)
            reward = 0.0

            # Reach: encourage gripper to approach cube (tanh for smooth gradient)
            gripper_to_cube = info["gripper_to_cube"]
            reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
            reward += reach_reward

            # Grasp bonus: cube pinched between gripper and jaw
            if info["is_grasping"]:
                reward += 0.25

            # Lift bonus: cube height above ground
            cube_z = info["cube_z"]
            if cube_z > 0.02:
                reward += 1.0

            # Success bonus
            if info["is_success"]:
                reward += 10.0

            return reward

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
