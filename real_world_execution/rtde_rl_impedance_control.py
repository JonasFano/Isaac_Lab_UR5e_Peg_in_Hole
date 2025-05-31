import sys
import time
import logging
import argparse
import math
import numpy as np
import torch
from stable_baselines3 import PPO
import csv

import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

from isaaclab.utils.math import compute_pose_error, quat_from_angle_axis


class RTDEJointPdExample:
    def __init__(self, robot_ip):
        logging.getLogger().setLevel(logging.INFO)
        # change this to match the robot's serial number
        self.timer_period = 0.002  # seconds

        # RL-related parameters
        self.noisy_hole_estimate = [0.5, -0.3, 0.0025] #[0.5, -0.3, 0.0025, 1.0, 0.0, 0.0, 0.0]
        self.previous_action = np.zeros(3)
        self.previous_tcp_pose = np.zeros(6)
        self.desired_ee_pose_b = torch.zeros(7)
        self.action_scaling = 0.004 # 0.0015

        # tcp_wrench_log.csv # K_p = [50, 50, 50, 100, 100, 100] # D_RATIO = 2 # action_scaling 0.02 # No video
        # tcp_wrench_log_v2.csv # K_p = [50, 50, 50, 100, 100, 100] # D_RATIO = 4 # action_scaling 0.02 # Video 3
        # tcp_wrench_log_v3.csv # K_p = [200, 200, 200, 400, 400, 400] # D_RATIO = 2 # action_scaling 0.02 # Video 4
        # tcp_wrench_log_v4.csv # K_p = [50, 50, 50, 100, 100, 100] # D_RATIO = 2 # action_scaling 0.02 # Video 5 # Successful
        # tcp_wrench_log_v5.csv # K_p = [50, 50, 100, 100, 100, 100] # D_RATIO = 2 # action_scaling 0.02 # Video 6 # Successful

        self.log_file = open("tcp_wrench_log_v14_5.csv", mode="w", newline="")
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(["timestamp", 
                                "tcp_x", "tcp_y", "tcp_z", 
                                "tcp_rx", "tcp_ry", "tcp_rz", 
                                "fx", "fy", "fz", "tx", "ty", "tz"])


        self.pose_error_b = torch.zeros(6)

        model_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/ur5e/impedance_ctrl/impedance_ctrl_peg_insert_2048_envs_v14/ckohomxv/model.zip"
        self.agent = PPO.load(model_path)

        self.q_err = np.array([0., 0., 0., 0., 0., 0.])
        self.time_counter = 0.
        self.initialized = False

        config_filename = "inv_dyn_configuration.xml"
        self.conf = rtde_config.ConfigFile(config_filename)
        state_names, state_types = self.conf.get_recipe("state")
        pose_error_b_names, pose_error_b_types = self.conf.get_recipe("pose_error_register")
        stamp_names, stamp_types = self.conf.get_recipe("control_stamp")

        RTDE_PORT = 30004
        self.con = rtde.RTDE(robot_ip, RTDE_PORT)
        self.con.connect()

        # get controller version
        self.con.get_controller_version()

        # setup recipes
        self.con.send_output_setup(state_names, state_types, frequency=50)
        self.pose_error_register = self.con.send_input_setup(pose_error_b_names, pose_error_b_types)
        self.control_stamp = self.con.send_input_setup(stamp_names, stamp_types)
        self.start_time = time.time()

        self.pose_error_register.input_double_register_0 = 0
        self.pose_error_register.input_double_register_1 = 0
        self.pose_error_register.input_double_register_2 = 0
        self.pose_error_register.input_double_register_3 = 0
        self.pose_error_register.input_double_register_4 = 0
        self.pose_error_register.input_double_register_5 = 0
        self.control_stamp.input_int_register_0 = 0
        self.control_stamp.input_double_register_6 = 0

        # start data synchronization
        if not self.con.send_start():
            sys.exit()

        state = self.con.receive()

        self.actual_TCP_pose = state.actual_TCP_pose
        self.actual_TCP_force = state.actual_TCP_force
        # self.ft_raw_wrench = state.ft_raw_wrench
        self.output_double_register_0 = state.output_double_register_0
        self.output_double_register_1 = state.output_double_register_1
        self.output_double_register_2 = state.output_double_register_2
        self.output_double_register_3 = state.output_double_register_3
        self.output_double_register_4 = state.output_double_register_4
        self.output_double_register_5 = state.output_double_register_5
        self.output_double_register_6 = state.output_double_register_6
        self.output_double_register_7 = state.output_double_register_7
        self.output_double_register_8 = state.output_double_register_8
        self.output_double_register_9 = state.output_double_register_9
        self.output_double_register_10 = state.output_double_register_10
        self.output_double_register_11 = state.output_double_register_11

        # send the initial torque values of 0
        self.con.send(self.pose_error_register)

    def axis_angle_to_quaternion(self, axis_angle):
        """
        Convert a single axis-angle representation to a quaternion (w, x, y, z),
        ensuring a consistent sign convention to prevent flipping.
        """
        if not isinstance(axis_angle, torch.Tensor):
            axis_angle = torch.tensor(axis_angle, dtype=torch.float32)

        angle = torch.norm(axis_angle)
        if angle < 1e-6:
            # Identity rotation
            quat_wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0], device=axis_angle.device)
        else:
            axis = axis_angle / angle
            quat_wxyz = quat_from_angle_axis(torch.tensor([angle], device=axis_angle.device), axis.unsqueeze(0))[0]  # (4,)

        return quat_wxyz

    def run_control_loop(self):
        try:
            while True:   
                state = self.con.receive()

                tcp_pose = torch.tensor(state.actual_TCP_pose)
                tcp_pos = tcp_pose[:3]
                tcp_quat = self.axis_angle_to_quaternion(tcp_pose[3:])
                self.tcp_pose = torch.cat((tcp_pos, tcp_quat), dim=0)
                self.wrench = np.array(state.actual_TCP_force)

                # print(self.tcp_pose)
                # print(self.wrench)

                self.observation = np.concatenate((self.tcp_pose[:3], self.wrench, self.noisy_hole_estimate, self.previous_action), axis=0)
                
                with torch.inference_mode():
                    self.action, _ = self.agent.predict(self.observation, deterministic=True)
                    self.action *= self.action_scaling 

                # print(self.action)
                action_tensor = torch.from_numpy(self.action).to(dtype=self.tcp_pose.dtype, device=self.tcp_pose.device)
                self.desired_ee_pose_b[:3] = (self.tcp_pose[:3] + action_tensor).to(self.desired_ee_pose_b)
                self.desired_ee_pose_b[3:] = torch.tensor([0.0, 0.0, 1.0, 0.0])

                # Compute pose error
                self.pose_error_b = torch.cat(
                    compute_pose_error(
                        self.tcp_pose[:3].unsqueeze(0),
                        self.tcp_pose[3:].unsqueeze(0),
                        self.desired_ee_pose_b[:3].unsqueeze(0),
                        self.desired_ee_pose_b[3:].unsqueeze(0),
                        rot_error_type="axis_angle",
                    ),
                    dim=-1,
                ).squeeze(0)

                print(self.pose_error_b)

                self.pose_error_register.input_double_register_0 = self.pose_error_b[0].item()
                self.pose_error_register.input_double_register_1 = self.pose_error_b[1].item()
                self.pose_error_register.input_double_register_2 = self.pose_error_b[2].item()
                self.pose_error_register.input_double_register_3 = self.pose_error_b[3].item()
                self.pose_error_register.input_double_register_4 = self.pose_error_b[4].item()
                self.pose_error_register.input_double_register_5 = self.pose_error_b[5].item()

    
                self.previous_tcp_pose = self.tcp_pose
                self.previous_action = self.action

                self.con.send(self.pose_error_register)

                self.csv_writer.writerow(
                    [time.time() - self.start_time] +            # timestamp
                    self.tcp_pose[:6].tolist() +                 # TCP position and orientation (axis-angle)
                    self.wrench.tolist()                         # Wrench [fx, fy, fz, tx, ty, tz]
                )


                # control_stamp is used to determine the timing between this control loop and the robot's control loop
                self.control_stamp.input_int_register_0 += 1
                if self.control_stamp.input_int_register_0 > 2147483647: # 2^31-1 to prevent overflow 
                    self.control_stamp.input_int_register_0 = 0
                self.control_stamp.input_double_register_6 = time.time() - self.start_time
                self.con.send(self.control_stamp)                
                    
        except KeyboardInterrupt:
            # Zero out all pose error registers
            self.pose_error_register.input_double_register_0 = 0.0
            self.pose_error_register.input_double_register_1 = 0.0
            self.pose_error_register.input_double_register_2 = 0.0
            self.pose_error_register.input_double_register_3 = 0.0
            self.pose_error_register.input_double_register_4 = 0.0
            self.pose_error_register.input_double_register_5 = 0.0
            self.con.send(self.pose_error_register)

            self.con.send_pause()
            self.con.disconnect()

            self.log_file.close()


def parse_args(args):
    parser = argparse.ArgumentParser(description="RTDE Torque Control Example")
    parser.add_argument(
        "-ip",
        "--robot_ip",
        dest="ip",
        help="IP address of the UR robot",
        type=str,
        default='localhost',
        metavar="<IP address of the UR robot>")
    return parser.parse_args(args)

def main(args):
    args = parse_args(args)
    pd_example = RTDEJointPdExample(args.ip)
    pd_example.run_control_loop()

if __name__ == '__main__':
    main(sys.argv[1:])
    
