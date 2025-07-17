# Import necessary modules and classes
import numpy as np
import mujoco
import mujoco.viewer
import time
from scipy.spatial.transform import Rotation as R
from dm_control.mujoco.wrapper import mjbindings
from dm_control import mjcf
mjlib = mjbindings.mjlib

np.set_printoptions(precision=4, suppress=True)

def task_space_inertia_matrix(M, J, threshold=1e-3):
    """Generate the task-space inertia matrix
    Parameters
    ----------
    M: np.array
        the generalized coordinates inertia matrix
    J: np.array
        the task space Jacobian
    threshold: scalar, optional (Default: 1e-3)
        singular value threshold, if the detminant of Mx_inv is less than
        this value then Mx is calculated using the pseudo-inverse function
        and all singular values < threshold * .1 are set = 0
    """

    # calculate the inertia matrix in task space
    M_inv = np.linalg.inv(M)
    Mx_inv = np.dot(J, np.dot(M_inv, J.T))
    if abs(np.linalg.det(Mx_inv)) >= threshold:
        # do the linalg inverse if matrix is non-singular
        # because it's faster and more accurate
        Mx = np.linalg.inv(Mx_inv)
    else:
        # using the rcond to set singular values < thresh to 0
        # singular values < (rcond * max(singular_values)) set to 0
        Mx = np.linalg.pinv(Mx_inv, rcond=threshold * 0.1)

    return Mx, M_inv

def pose_error(target_pose, ee_pose) -> np.ndarray:
    """
    Calculate the rotational error (orientation difference) between the target and current orientation.

    Parameters:
        target_ori_mat (numpy.ndarray): The target orientation matrix.
        current_ori_mat (numpy.ndarray): The current orientation matrix.

    Returns:
        numpy.ndarray: The rotational error in axis-angle representation.
    """
    target_pos = target_pose[:3]
    target_quat = target_pose[3:]
    ee_pos = ee_pose[:3]
    ee_quat = ee_pose[3:]

    err_pos = target_pos - ee_pos
    err_ori = orientation_error(R.from_quat(target_quat).as_matrix(), R.from_quat(ee_quat).as_matrix())

    return np.concatenate([err_pos, err_ori])

def orientation_error(desired, current):
    """
    This function calculates a 3-dimensional orientation error vector for use in the
    impedance controller. It does this by computing the delta rotation between the
    inputs and converting that rotation to exponential coordinates (axis-angle
    representation, where the 3d vector is axis * angle).
    See https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation for more information.
    Optimized function to determine orientation error from matrices

    Args:
        desired (np.array): 2d array representing target orientation matrix
        current (np.array): 2d array representing current orientation matrix

    Returns:
        np.array: 2d array representing orientation error as a matrix
    """
    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]

    error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))

    return error

def get_site_jac(model, data, site_id):
    """Return the Jacobian' translational component of the end-effector of
    the corresponding site id.
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mjlib.mj_jacSite(model, data, jacp, jacr, site_id)
    jac = np.vstack([jacp, jacr])

    return jac

def get_fullM(model, data):
    M = np.zeros((model.nv, model.nv))
    mjlib.mj_fullM(model, M, data.qM)

    return M

class JointEffortController:
    def __init__(
        self,
        physics,
        joints,
        min_effort: np.ndarray,
        max_effort: np.ndarray,
    ) -> None:

        self._physics = physics
        self._joints = joints
        self._min_effort = min_effort
        self._max_effort = max_effort

    def run(self, target) -> None:
        """
        Run the robot controller.

        Parameters:
            target (numpy.ndarray): The desired target joint positions or states for the robot.
                                   The size of `target` should be (n_joints,) where n_joints is the number of robot joints.
            ctrl (numpy.ndarray): Control signals for the robot actuators from `mujoco._structs.MjData.ctrl` of size (nu,).
        """

        # Clip the target efforts to ensure they are within the allowable effort range
        target_effort = np.clip(target, self._min_effort, self._max_effort)
        # Set the control signals for the actuators to the desired target joint positions or states
        self._physics.bind(self._joints).qfrc_applied = target_effort

    def reset(self) -> None:
        pass

class OperationalSpaceController(JointEffortController):
    def __init__(
        self,
        physics,
        joints,
        eef_site,
        min_effort: np.ndarray,
        max_effort: np.ndarray,
        kp: float,
        ko: float,
        kv: float,
        vmax_xyz: float,
        vmax_abg: float,
    ) -> None:
       
        super().__init__(physics, joints, min_effort, max_effort)

        self._eef_site = eef_site
        self._kp = kp
        self._ko = ko
        self._kv = kv
        self._vmax_xyz = vmax_xyz
        self._vmax_abg = vmax_abg

        self._eef_id = self._physics.bind(eef_site).element_id
        self._jnt_dof_ids = self._physics.bind(joints).dofadr
        self._dof = len(self._jnt_dof_ids)

        self._task_space_gains = np.array([self._kp] * 3 + [self._ko] * 3)
        self._lamb = self._task_space_gains / self._kv
        self._sat_gain_xyz = vmax_xyz / self._kp * self._kv
        self._sat_gain_abg = vmax_abg / self._ko * self._kv
        self._scale_xyz = vmax_xyz / self._kp * self._kv
        self._scale_abg = vmax_abg / self._ko * self._kv

    def run(self, target):
        # target pose is a 7D vector [x, y, z, qx, qy, qz, qw]
        target_pose = target

        # Get the Jacobian matrix for the end-effector.
        J = get_site_jac(
            self._physics.model.ptr,
            self._physics.data.ptr,
            self._eef_id,
        )
        J = J[:, self._jnt_dof_ids]
        # Get the mass matrix and its inverse for the controlled degrees of freedom (DOF) of the robot.
        M_full = get_fullM(
            self._physics.model.ptr,
            self._physics.data.ptr,
        )
        M = M_full[self._jnt_dof_ids, :][:, self._jnt_dof_ids]
        Mx, M_inv = task_space_inertia_matrix(M, J)

        # Get the joint velocities for the controlled DOF.
        dq = self._physics.bind(self._joints).qvel

        # Get the end-effector position, orientation matrix, and twist (spatial velocity).
        ee_pos = self._physics.bind(self._eef_site).xpos
        ee_quat = R.from_matrix(self._physics.bind(self._eef_site).xmat.reshape(3, 3)).as_quat()
        ee_pose = np.concatenate([ee_pos, ee_quat])

        # Calculate the pose error (difference between the target and current pose).
        pose_err = pose_error(target_pose, ee_pose)
        
        # Initialize the task space control signal (desired end-effector motion).
        u_task = np.zeros(6)

        # Calculate the task space control signal.
        u_task += self._scale_signal_vel_limited(pose_err)

        # joint space control signal
        u = np.zeros(self._dof)
       
        # Add the task space control signal to the joint space control signal
        u += np.dot(J.T, np.dot(Mx, u_task))

        # Add damping to joint space control signal
        u += -self._kv * np.dot(M, dq)

        # Add gravity compensation to the target effort
        u += self._physics.bind(self._joints).qfrc_bias

        # send the target effort to the joint effort controller
        super().run(u)

    def _scale_signal_vel_limited(self, u_task: np.ndarray) -> np.ndarray:
        """
        Scale the control signal such that the arm isn't driven to move faster in position or orientation than the specified vmax values.

        Parameters:
            u_task (numpy.ndarray): The task space control signal.

        Returns:
            numpy.ndarray: The scaled task space control signal.
        """
        norm_xyz = np.linalg.norm(u_task[:3])
        norm_abg = np.linalg.norm(u_task[3:])
        scale = np.ones(6)
        if norm_xyz > self._sat_gain_xyz:
            scale[:3] *= self._scale_xyz / norm_xyz
        if norm_abg > self._sat_gain_abg:
            scale[3:] *= self._scale_abg / norm_abg

        return self._kv * scale * self._lamb * u_task


def set_mocap_pose(mocap, physics, position=None, quaternion=None):
    """
    Sets the pose of the mocap body.

    Args:
        physics: The physics simulation.
        position: The position of the mocap body.
        quaternion: The quaternion orientation of the mocap body.
    """

    # flip quaternion xyzw to wxyz
    quaternion = np.roll(np.array(quaternion), 1)

    if position is not None:
        physics.bind(mocap).mocap_pos[:] = position
    if quaternion is not None:
        physics.bind(mocap).mocap_quat[:] = quaternion

def get_mocap_pose(mocap, physics):
    position = physics.bind(mocap).mocap_pos[:]
    quaternion = physics.bind(mocap).mocap_quat[:]

    # flip quaternion wxyz to xyzw
    quaternion = np.roll(np.array(quaternion), -1)

    pose = np.concatenate([position, quaternion])

    return pose

def main() -> None:
    model = mjcf.from_path("universal_robots_ur10e/scene.xml")
    mocap = model.worldbody.add("body", name="mocap", mocap=True)
    mocap.add(
            "geom",
            type="box",
            size=[0.015] * 3,
            rgba=[1, 0, 0, 0.2],
            conaffinity=0,
            contype=0,
        )
    physics = mjcf.Physics.from_mjcf_model(model)

    physics.bind(model.find_all('joint')).qpos = [
        0.0,
        -1.5707,
        1.5707,
        -1.5707,
        -1.5707,
        0.0,
    ]
    set_mocap_pose(mocap, physics, position=[0.5, 0, 0.3], quaternion=[0, 0, 0, 1])
    
    controller = OperationalSpaceController(
                physics=physics,
                joints=model.find_all('joint'),
                eef_site=model.find('site', 'eef_site'),
                min_effort=-150.0,
                max_effort=150.0,
                kp=200,
                ko=200,
                kv=50,
                vmax_xyz=1.0,
                vmax_abg=2.0,
            )
   
    timestep = 0.002
    with mujoco.viewer.launch_passive(
        physics.model.ptr, physics.data.ptr, show_left_ui=False, show_right_ui=False
    ) as viewer:
        while viewer.is_running():
            step_start = time.time()

            target_pose = get_mocap_pose(mocap, physics)
            controller.run(target_pose)
            physics.step()
            viewer.sync()

            time_until_next_step = timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    viewer.close()

if __name__ == "__main__":
    main()

