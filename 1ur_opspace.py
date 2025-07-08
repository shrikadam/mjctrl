import mujoco
import mujoco.viewer
import numpy as np
import time

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 1.0

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Maximum allowable joint velocity in rad/s. Set to 0 to disable.
max_angvel = 0.0

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("universal_robots_ur5e/scene.xml")
    data = mujoco.MjData(model)

    # Override the simulation timestep.
    model.opt.timestep = dt

    # End-effector site we wish to control, in this case a site attached to the last
    # link (wrist_3_link) of the robot.
    site_id = model.site("attachment_site").id

    # Name of bodies we wish to apply gravity compensation to.
    body_names = [
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
    ]
    body_ids = [model.body(name).id for name in body_names]
    if gravity_compensation:
        model.body_gravcomp[body_ids] = 1.0

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
    # Note that actuator names are the same as joint names in this case.
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    # Initial joint configuration saved as a keyframe in the XML file.
    key_id = model.key("home").id
    
    # Mocap body we will control with our mouse.
    mocap_id = model.body("target").mocapid[0]

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

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

            # Get the Jacobian with respect to the end-effector site.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            ################################################################
            # # Solve system of equations: J @ dq = error.
            # dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

            # # Scale down joint velocities if they exceed maximum.
            # if max_angvel > 0:
            #     dq_abs_max = np.abs(dq).max()
            #     if dq_abs_max > max_angvel:
            #         dq *= max_angvel / dq_abs_max

            # # Integrate joint velocities to obtain joint positions.
            # q = data.qpos.copy()
            # mujoco.mj_integratePos(model, q, dq, integration_dt)

            # # Set the control signal.
            # np.clip(q, *model.jnt_range.T, out=q)
            ################################################################
            kp = 200.0
            ko = 200.0
            kv = 50.0
            min_effort = -150.0
            max_effort = 150.0
            vmax_xyz = 1.0
            vmax_abg = 2.0
            
            task_space_gains = np.array([kp] * 3 + [ko] * 3)
            lamb = task_space_gains / kv
            sat_gain_xyz = vmax_xyz / kp * kv
            sat_gain_abg = vmax_abg / ko * kv
            scale_xyz = vmax_xyz / kp * kv
            scale_abg = vmax_abg / ko * kv

            M_full = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, M_full, data.qM)

            M_inv = np.zeros((model.nv, model.nv))
            # Compute the task-space inertia matrix.
            mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
            Mx_inv = jac @ M_inv @ jac.T
            if abs(np.linalg.det(Mx_inv)) >= 1e-2:
                Mx = np.linalg.inv(Mx_inv)
            else:
                Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

            dq = data.qvel

            u_task = np.zeros(6)
            norm_xyz = np.linalg.norm(u_task[:3])
            norm_abg = np.linalg.norm(u_task[3:])
            scale = np.ones(6)
            if norm_xyz > sat_gain_xyz:
                scale[:3] *= scale_xyz / norm_xyz
            if norm_abg > sat_gain_abg:
                scale[3:] *= scale_abg / norm_abg

            u_task += kv * scale * lamb * u_task
            u = np.zeros(model.nv)
            # Add the task space control signal to the joint space control signal
            u += np.dot(jac.T, np.dot(Mx, u_task))
            # Add damping to joint space control signal
            u += -kv * np.dot(M_full, dq)
            # Add gravity compensation to the target effort
            u += data.qfrc_bias

            np.clip(u, *model.actuator_ctrlrange.T, out=u)
            # qcmd = np.clip(u, min_effort, max_effort)
            # data.qfrc_applied = qcmd

            # # Compute generalized forces.
            # tau = jac.T @ Mx @ (Kp * error - Kd * (jac @ data.qvel[dof_ids]))

            # # Add joint task in nullspace.
            # Jbar = M_inv @ jac.T @ Mx
            # ddq = Kp_null * (q0 - data.qpos[dof_ids]) - Kd_null * data.qvel[dof_ids]
            # tau += (np.eye(model.nv) - jac.T @ Jbar.T) @ ddq

            # # Add gravity compensation.
            # if gravity_compensation:
            #     tau += data.qfrc_bias[dof_ids]

            # Set the control signal and step the simulation.
            # np.clip(tau, *model.actuator_ctrlrange.T, out=tau)
            ################################################################
            data.ctrl[actuator_ids] = u[dof_ids]
            
            # Step the simulation.
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
