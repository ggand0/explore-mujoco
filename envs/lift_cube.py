"""Lift cube environment - intermediate curriculum task for SO-101 arm."""
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class LiftCubeEnv(gym.Env):
    """Environment for lifting a cube above a height threshold.

    This is a simpler task than pick-and-place, designed to teach the agent
    to grasp before learning placement. Pushing cannot solve this task.

    Observation space (18 dims):
        - Joint positions (6)
        - Joint velocities (6)
        - Gripper position (3)
        - Cube position (3)

    Action space (6 dims):
        - Delta joint positions for all 6 joints (continuous)

    Success condition:
        - Cube z-position > lift_height for hold_steps consecutive steps
        - While gripper is in contact with cube (grasping)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        action_scale: float = 0.1,
        lift_height: float = 0.08,
        hold_steps: int = 10,
        reward_type: str = "sparse",
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

        # Joint info
        self.n_joints = 6
        self.ctrl_ranges = self.model.actuator_ctrlrange.copy()

        # Action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        # Observation space (no target position needed for lift task)
        obs_dim = 6 + 6 + 3 + 3  # joints pos + vel + gripper pos + cube pos
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Renderer
        self._renderer = None
        if render_mode == "human":
            self._renderer = mujoco.Renderer(self.model)

    def _get_obs(self) -> np.ndarray:
        joint_pos = self.data.qpos[: self.n_joints].copy()
        joint_vel = self.data.qvel[: self.n_joints].copy()
        gripper_pos = self.data.sensor("gripper_pos").data.copy()
        cube_pos = self.data.sensor("cube_pos").data.copy()

        return np.concatenate([joint_pos, joint_vel, gripper_pos, cube_pos]).astype(np.float32)

    def _get_gripper_state(self) -> float:
        gripper_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper"
        )
        gripper_qpos_addr = self.model.jnt_qposadr[gripper_joint_id]
        return self.data.qpos[gripper_qpos_addr]

    def _has_cube_contact(self) -> bool:
        cube_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"
        )
        gripper_geom_ids = set(range(25, 31))

        for i in range(self.data.ncon):
            geom1 = self.data.contact[i].geom1
            geom2 = self.data.contact[i].geom2
            if geom1 == cube_geom_id and geom2 in gripper_geom_ids:
                return True
            if geom2 == cube_geom_id and geom1 in gripper_geom_ids:
                return True
        return False

    def _is_grasping(self) -> bool:
        gripper_state = self._get_gripper_state()
        # Gripper range is ~0.194 (closed) to ~0.324 (open)
        # Use midpoint as threshold - must be clearly closed
        is_closed = gripper_state < 0.25
        has_contact = self._has_cube_contact()
        return is_closed and has_contact

    def _get_info(self) -> dict[str, Any]:
        gripper_pos = self.data.sensor("gripper_pos").data.copy()
        cube_pos = self.data.sensor("cube_pos").data.copy()

        gripper_to_cube = np.linalg.norm(gripper_pos - cube_pos)
        cube_z = cube_pos[2]
        is_grasping = self._is_grasping()
        is_lifted = is_grasping and cube_z > self.lift_height

        return {
            "gripper_to_cube": gripper_to_cube,
            "cube_pos": cube_pos.copy(),
            "cube_z": cube_z,
            "gripper_pos": gripper_pos.copy(),
            "gripper_state": self._get_gripper_state(),
            "has_contact": self._has_cube_contact(),
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

        self._step_count = 0
        self._hold_count = 0

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Apply action
        action = np.clip(action, -1.0, 1.0)
        delta = action * self.action_scale
        current_ctrl = self.data.ctrl.copy()
        new_ctrl = current_ctrl + delta
        new_ctrl = np.clip(new_ctrl, self.ctrl_ranges[:, 0], self.ctrl_ranges[:, 1])
        self.data.ctrl[:] = new_ctrl

        # Step simulation
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
            # -1 until success, then 0
            return 0.0 if info["is_success"] else -1.0
        else:
            # Dense reward
            reward = 0.0

            # Reach: encourage gripper to approach cube
            gripper_to_cube = info["gripper_to_cube"]
            reward -= gripper_to_cube

            # Lift bonus: cube height above ground (only way to get this is to grasp)
            # No separate grasp bonus - if cube is high, it must be grasped
            cube_z = info["cube_z"]
            if cube_z > 0.02:  # above resting height
                reward += cube_z * 20.0

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
