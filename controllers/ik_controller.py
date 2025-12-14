"""Inverse Kinematics controller for SO-101 arm using MuJoCo's Jacobian."""
import mujoco
import numpy as np


class IKController:
    """Damped least-squares IK controller.

    Uses MuJoCo's mj_jac to compute the Jacobian and solve for joint velocities
    that move the end-effector toward a target position.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        end_effector_site: str = "gripperframe",
        damping: float = 0.1,
        max_dq: float = 0.5,
    ):
        """Initialize IK controller.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            end_effector_site: Name of the end-effector site
            damping: Damping factor for singularity robustness
            max_dq: Maximum joint velocity per step
        """
        self.model = model
        self.data = data
        self.damping = damping
        self.max_dq = max_dq

        # Get end-effector site ID
        self.ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, end_effector_site
        )
        if self.ee_site_id == -1:
            raise ValueError(f"Site '{end_effector_site}' not found in model")

        # Get number of arm joints (excluding gripper)
        # SO-101: 5 arm joints + 1 gripper
        self.n_arm_joints = 5
        self.n_total_joints = model.nv

        # Pre-allocate Jacobians
        self.jacp = np.zeros((3, self.n_total_joints))  # position Jacobian
        self.jacr = np.zeros((3, self.n_total_joints))  # rotation Jacobian

    def get_ee_position(self) -> np.ndarray:
        """Get current end-effector position."""
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_ee_orientation(self) -> np.ndarray:
        """Get current end-effector orientation as rotation matrix."""
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

    def compute_joint_velocities(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute joint velocities to move end-effector toward target.

        Args:
            target_pos: Target position (3,)
            target_quat: Target orientation as quaternion (4,), optional

        Returns:
            Joint velocities (n_arm_joints,)
        """
        # Get current end-effector position
        current_pos = self.get_ee_position()

        # Position error
        pos_error = target_pos - current_pos

        # Compute Jacobian at current end-effector position
        mujoco.mj_jacSite(
            self.model, self.data,
            self.jacp, self.jacr,
            self.ee_site_id
        )

        # Use only arm joints (first 5), not gripper
        J = self.jacp[:, :self.n_arm_joints]

        if target_quat is not None:
            # Include orientation error (simplified - just position for now)
            # TODO: Add orientation control if needed
            pass

        # Damped least-squares pseudoinverse: (J^T J + λ²I)^-1 J^T
        # More stable than pure pseudoinverse near singularities
        JTJ = J.T @ J
        damping_matrix = self.damping**2 * np.eye(self.n_arm_joints)

        try:
            dq = np.linalg.solve(JTJ + damping_matrix, J.T @ pos_error)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if solve fails
            dq = np.linalg.pinv(J) @ pos_error

        # Clamp to max velocity
        dq = np.clip(dq, -self.max_dq, self.max_dq)

        return dq

    def step_toward_target(
        self,
        target_pos: np.ndarray,
        gripper_action: float = 0.0,
        gain: float = 1.0,
    ) -> np.ndarray:
        """Compute control signal to move toward target position.

        Args:
            target_pos: Target end-effector position (3,)
            gripper_action: Gripper control (-1 to 1, mapped to control range)
            gain: Proportional gain for velocity

        Returns:
            Full control vector (6,) for all actuators
        """
        # Compute arm joint velocities
        dq = self.compute_joint_velocities(target_pos)
        dq *= gain

        # Current arm joint positions
        current_q = self.data.qpos[:self.n_arm_joints].copy()

        # Target joint positions
        target_q = current_q + dq

        # Clamp to joint limits
        for i in range(self.n_arm_joints):
            jnt_range = self.model.jnt_range[i]
            if jnt_range[0] != jnt_range[1]:  # has limits
                target_q[i] = np.clip(target_q[i], jnt_range[0], jnt_range[1])

        # Build full control vector
        ctrl = np.zeros(self.model.nu)
        ctrl[:self.n_arm_joints] = target_q

        # Gripper control (map -1..1 to control range)
        gripper_range = self.model.actuator_ctrlrange[5]
        gripper_ctrl = (gripper_action + 1) / 2 * (gripper_range[1] - gripper_range[0]) + gripper_range[0]
        ctrl[5] = gripper_ctrl

        return ctrl


def demo():
    """Demo the IK controller with interactive target."""
    from pathlib import Path
    import mujoco.viewer

    scene_path = Path(__file__).parent.parent / "SO-ARM100/Simulation/SO101/lift_cube_scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    ik = IKController(model, data)

    # Target position (above the cube)
    target = np.array([0.40, -0.10, 0.15])

    print(f"Initial EE position: {ik.get_ee_position()}")
    print(f"Target position: {target}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Compute and apply control
            ctrl = ik.step_toward_target(target, gripper_action=0.0, gain=0.5)
            data.ctrl[:] = ctrl

            # Step simulation
            mujoco.mj_step(model, data)

            # Update viewer
            viewer.sync()

            # Print progress
            current = ik.get_ee_position()
            error = np.linalg.norm(target - current)
            if error < 0.01:
                print(f"Reached target! Error: {error:.4f}")


if __name__ == "__main__":
    demo()
