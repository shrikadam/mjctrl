import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

np.set_printoptions(suppress=True, precision=3)

# --- Helper Functions ---
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

def site_jac(model, data, site_id):
    """Return the Jacobian' translational component of the end-effector of
    the corresponding site id.
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    jac = np.vstack([jacp, jacr])

    return jac

def full_inertia_mat(model, data):
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)

    return M

def task_space_inertia_mat(M, J, threshold=1e-3):
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

    return Mx

# --- The Robust Controller Class ---
class OperationalSpaceController:
    def __init__(self, model, data, site_id, dof_ids, kp, ko, kv, vmax_xyz, vmax_abg, min_effort, max_effort):
        self.model = model
        self.data = data
        self.site_id = site_id
        self.dof_ids = dof_ids
        
        # Gains and Limits
        self.kp = kp
        self.ko = ko
        self.kv = kv
        self.vmax_xyz = vmax_xyz
        self.vmax_abg = vmax_abg
        
        # Precompute gain ratios for saturation
        self.sat_gain_xyz = vmax_xyz / kp * kv
        self.sat_gain_abg = vmax_abg / ko * kv
        self.scale_xyz = vmax_xyz / kp * kv
        self.scale_abg = vmax_abg / ko * kv
        self._lamb = np.array([self.kp] * 3 + [self.ko] * 3) / self.kv

        self.min_effort = min_effort
        self.max_effort = max_effort

    def run(self, target_pose):
        # 1. Get Jacobian and Mass Matrix
        J_full = site_jac(self.model, self.data, self.site_id)
        
        J = J_full[:, self.dof_ids] # Filter for controlled joints
        
        M_full = full_inertia_mat(self.model, self.data)

        M = M_full[self.dof_ids, :][:, self.dof_ids] # Filter Mass Matrix
        
        # 2. Compute Task-Space Inertia (Mx)
        Mx = task_space_inertia_mat(M, J)

        # 3. Get Current State
        dq = self.data.qvel[self.dof_ids]
        ee_pos = self.data.site_xpos[self.site_id]
        ee_quat = R.from_matrix(self.data.site_xmat[self.site_id].reshape(3, 3)).as_quat()
        current_pose = np.concatenate([ee_pos, ee_quat])

        # 4. Compute Pose Error (Passing the Rotation Matrix now)
        pose_err = pose_error(target_pose, current_pose)

        # 5. Compute Task Space Control Signal (u_task) with Velocity Saturation
        norm_xyz = np.linalg.norm(pose_err[:3])
        norm_abg = np.linalg.norm(pose_err[3:])
        scale = np.ones(6)
        if norm_xyz > self.sat_gain_xyz:
            scale[:3] *= self.scale_xyz / norm_xyz
        if norm_abg > self.sat_gain_abg:
            scale[3:] *= self.scale_abg / norm_abg
            
        u_task = self.kv * scale * self._lamb * pose_err 

        # 6. Compute Joint Torques
        tau = J.T @ (Mx @ u_task)

        # 7. Add Joint Damping 
        tau += -self.kv * (M @ dq)

        # 8. Add Gravity Compensation
        tau += self.data.qfrc_bias[self.dof_ids]

        # 9. Clip the target efforts to within allowable range
        tau = np.clip(tau, self.min_effort, self.max_effort)

        return tau

# --- Main Simulation Loop ---
def main():
    xml_path = "universal_robots_ur10e/scene.xml" 
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    site_name = "attachment_site"
    site_id = model.site(site_name).id
    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]
    
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow", 
                   "wrist_1", "wrist_2", "wrist_3"]
    dof_ids = [model.joint(name).id for name in joint_names]

    controller = OperationalSpaceController(
        model, data, site_id, dof_ids,
        kp=200, ko=200, kv=50, vmax_xyz=1.0, vmax_abg=2.0, min_effort=-150.0, max_effort=150.0,
    )

    init_q = [0, -1.57, 1.57, -1.57, -1.57, 0]
    data.qpos[dof_ids] = init_q
    mujoco.mj_forward(model, data)
    
    # Initialize Target
    # data.mocap_pos[mocap_id] = data.site_xpos[site_id]
    # mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], data.site_xmat[site_id])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            target_pos = data.mocap_pos[mocap_id]
            target_quat = data.mocap_quat[mocap_id]
            # Flip quaternion wxyz to xyzw
            target_quat = np.roll(np.array(target_quat), -1)
            target_pose = np.concatenate([target_pos, target_quat])
            
            tau = controller.run(target_pose)
            
            data.qfrc_applied[dof_ids] = tau
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()