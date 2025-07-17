import mujoco
import mujoco.viewer
import numpy as np
import time

min_effort = -150.0
max_effort = 150.0
kp = 200.0
ko = 200.0
kv = 50.0
vmax_xyz = 1.0
vmax_abg = 2.0
task_space_gains = np.array([kp] * 3 + [ko] * 3)
lamb = task_space_gains / kv
sat_gain_xyz = vmax_xyz / kp * kv
sat_gain_abg = vmax_abg / ko * kv
scale_xyz = vmax_xyz / kp * kv
scale_abg = vmax_abg / ko * kv
dt: float = 0.002


np.set_printoptions(precision=4, suppress=True)

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("universal_robots_ur10e/scene.xml")
    data = mujoco.MjData(model)
    
    # Override the simulation timestep.
    model.opt.timestep = dt

    # End-effector site we wish to control, in this case a site attached to the last
    # link (wrist_3_link) of the robot.
    site_name = "eef_site"
    site_id = model.site(site_name).id

    # Get the dof and actuator ids for the joints we wish to control.
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    # actuator_ids = np.array([model.actuator(name).id for name in joint_names])
    
    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    
    # Mocap body we will control with our mouse.
    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv = np.zeros((model.nv, model.nv))
    M_full = np.zeros((model.nv, model.nv))

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Reset the simulation to the initial keyframe.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Initialize the camera view to that of the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Toggle site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()

            # Position error.
            error_pos[:] = data.mocap_pos[mocap_id] - data.site(site_id).xpos

            # Orientation error.
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)
            mujoco.mj_forward(model, data)
            # Get the Jacobian with respect to the end-effector site.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            # Calculate full inertia matrix
            mujoco.mj_fullM(model, M_full, data.qM)
            
            # Calculate the task space inertia matrix 
            M_inv = np.linalg.inv(M_full)
            Mx_inv = np.dot(jac, np.dot(M_inv, jac.T))
            if abs(np.linalg.det(Mx_inv)) >= 1e-3:
                # do the linalg inverse if matrix is non-singular
                # because it's faster and more accurate
                Mx = np.linalg.inv(Mx_inv)
            else:
                # using the rcond to set singular values < thresh to 0
                # singular values < (rcond * max(singular_values)) set to 0
                Mx = np.linalg.pinv(Mx_inv, rcond=1e-3 * 0.1)
            
            dq = data.qvel[dof_ids]

            # Initialize the task space control signal (desired end-effector motion).
            u_task = np.zeros(6)

            norm_xyz = np.linalg.norm(error_pos)
            norm_abg = np.linalg.norm(error_ori)
            scale = np.ones(6)
            if norm_xyz > sat_gain_xyz:
                scale[:3] *= scale_xyz / norm_xyz
            if norm_abg > sat_gain_abg:
                scale[3:] *= scale_abg / norm_abg

            u_task += kv * scale * lamb * error
            # joint space control signal
            u = np.zeros(model.nv)
            # Add the task space control signal to the joint space control signal
            u += np.dot(jac.T, np.dot(Mx, u_task))
            # Add damping to joint space control signal
            u += -kv * np.dot(M_full, dq)
            # Add gravity compensation to the target effort
            u += data.qfrc_bias[dof_ids]
            # Clip the target efforts to ensure they are within the allowable effort range
            target_effort = np.clip(u, min_effort, max_effort)
            # Set the control signals for the actuators to the desired target joint positions or states
            data.qfrc_applied[dof_ids] = target_effort
            # data.ctrl[actuator_ids] = target_effort[actuator_ids]
            
            # Step the simulation.
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()