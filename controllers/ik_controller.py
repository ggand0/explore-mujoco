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

    def _quat_to_mat(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
        w, x, y, z = quat
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

    def _orientation_error(self, target_mat: np.ndarray, current_mat: np.ndarray) -> np.ndarray:
        """Compute orientation error as axis-angle vector.

        Returns a 3D vector whose direction is the rotation axis and
        magnitude is the rotation angle (in radians).
        """
        # Relative rotation: R_error = R_target @ R_current^T
        R_error = target_mat @ current_mat.T

        # Convert to axis-angle using Rodrigues formula
        # trace(R) = 1 + 2*cos(theta)
        trace = np.trace(R_error)
        cos_theta = (trace - 1) / 2
        cos_theta = np.clip(cos_theta, -1, 1)
        theta = np.arccos(cos_theta)

        if theta < 1e-6:
            return np.zeros(3)

        # axis = (R - R^T) / (2*sin(theta)) as skew-symmetric
        axis = np.array([
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1]
        ]) / (2 * np.sin(theta))

        return axis * theta

    def compute_joint_velocities(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        orientation_weight: float = 1.0,
        locked_joints: list[int] | None = None,
    ) -> np.ndarray:
        """Compute joint velocities to move end-effector toward target.

        Args:
            target_pos: Target position (3,)
            target_quat: Target orientation as quaternion (w,x,y,z), optional
            orientation_weight: Weight for orientation error vs position error
            locked_joints: List of joint indices (0-4) to exclude from IK

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

        # Determine which joints to use
        if locked_joints is None:
            active_joints = list(range(self.n_arm_joints))
        else:
            active_joints = [i for i in range(self.n_arm_joints) if i not in locked_joints]

        n_active = len(active_joints)

        # Use only active joints
        Jp = self.jacp[:, active_joints]
        Jr = self.jacr[:, active_joints]

        if target_quat is not None:
            # Combined position + orientation IK
            target_mat = self._quat_to_mat(target_quat)
            current_mat = self.get_ee_orientation()
            ori_error = self._orientation_error(target_mat, current_mat) * orientation_weight

            # Stack position and orientation
            error = np.concatenate([pos_error, ori_error])
            J = np.vstack([Jp, Jr])
        else:
            # Position-only IK
            error = pos_error
            J = Jp

        # Damped least-squares pseudoinverse: (J^T J + λ²I)^-1 J^T
        # More stable than pure pseudoinverse near singularities
        JTJ = J.T @ J
        damping_matrix = self.damping**2 * np.eye(n_active)

        try:
            dq_active = np.linalg.solve(JTJ + damping_matrix, J.T @ error)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if solve fails
            dq_active = np.linalg.pinv(J) @ error

        # Clamp to max velocity
        dq_active = np.clip(dq_active, -self.max_dq, self.max_dq)

        # Map back to full joint vector
        dq = np.zeros(self.n_arm_joints)
        for i, joint_idx in enumerate(active_joints):
            dq[joint_idx] = dq_active[i]

        return dq

    def step_toward_target(
        self,
        target_pos: np.ndarray,
        gripper_action: float = 0.0,
        gain: float = 1.0,
        target_quat: np.ndarray | None = None,
        orientation_weight: float = 1.0,
        locked_joints: list[int] | None = None,
    ) -> np.ndarray:
        """Compute control signal to move toward target position and orientation.

        Args:
            target_pos: Target end-effector position (3,)
            gripper_action: Gripper control (-1 to 1, mapped to control range)
            gain: Proportional gain for velocity
            target_quat: Target orientation as quaternion (w,x,y,z), optional
            orientation_weight: Weight for orientation error vs position error
            locked_joints: List of joint indices (0-4) to exclude from IK

        Returns:
            Full control vector (6,) for all actuators
        """
        # Compute arm joint velocities
        dq = self.compute_joint_velocities(target_pos, target_quat, orientation_weight, locked_joints)
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
