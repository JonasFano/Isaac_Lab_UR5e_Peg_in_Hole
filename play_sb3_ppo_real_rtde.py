import os
import csv
import random
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO
import rtde_control
import rtde_receive

class UR5eRobotController:
    """
    Class to control a UR5e robot using RTDE interface and a pre-trained PPO model.
    """

    def __init__(self, robot_ip, model_path, action_scaling, save_dir, filename, mode, save=False):
        # Initialize RTDE Control and Receive Interfaces
        self.robot_ip = robot_ip
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        print("Connected to Interfaces")

        # Load the pre-trained model
        print(f"Loading checkpoint from: {model_path}")
        self.agent = PPO.load(model_path)

        # Set CSV save directory
        self.save_dir = save_dir
        self.csv_path = os.path.join(save_dir, filename)

        self.save = save
        self.action_scaling = action_scaling
        self.mode = mode
        self.current_index = 0
        self.prev_quaternion = None
        self.prev_rotvec = None
        self.prev_rotvec_2 = None
        self.previous_action = np.zeros(6)

        # 180-degree rotation around Z-axis
        self.ROT_180_Z = R.from_euler('z', 180, degrees=True)


    def save_observations_to_csv(self, timestep):
        """Save TCP pose, target pose, and last action to a CSV file."""
        header = (
            ["timestep"]
            + [f"tcp_pose_{i}" for i in range(7)]
            + [f"target_pose_{i}" for i in range(7)]
            + [f"last_action_{i}" for i in range(6)]
        )

        # Flatten all observations into a single list
        data = [timestep] + self.tcp_pose.tolist() + self.target_pose.tolist() + self.previous_action.tolist()

        file_exists = os.path.isfile(self.csv_path)

        with open(self.csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)  # Write header if file doesn't exist
            writer.writerow(data)


    def move_robot_to_home(self):
        """Move robot to home pose specified with joint positions."""
        home_pose = np.array([1.3, -2.0, 2.0, -1.5, -1.5, 0.0]) # 3.14]) # 0.0]) 
        speed = 1.5
        acceleration = 1.5
        self.rtde_c.moveJ(home_pose, speed=speed, acceleration=acceleration)


    def sample_random_pose(self):
        """Randomly sample a target pose within specified bounds."""
        pos_x = random.uniform(-0.2, 0.2)
        pos_y = random.uniform(-0.45, -0.25)
        pos_z = random.uniform(0.2, 0.5)
        # pos_x = random.uniform(0.1, 0.1)
        # pos_y = random.uniform(-0.3, -0.3)
        # pos_z = random.uniform(0.3, 0.3)
        roll = 0.0
        pitch = np.pi  # End-effector z-axis pointing down (180 deg rotation)
        yaw = random.uniform(-2.5*np.pi, -1.5*np.pi) # For wrist_3_joint = 0.0
        # yaw = random.uniform(-3/2*np.pi, -1/3*np.pi) # For wrist_3_joint = 3.14
        # yaw = random.uniform(-4/3*np.pi, -4/3*np.pi) # Target Orientation not rotated: [-3.06161700e-17  8.66025404e-01 -5.00000000e-01 -5.30287619e-17] # Rotated: [ 5.30287619e-17  5.00000000e-01  8.66025404e-01 -3.06161700e-17]
        # yaw = random.uniform(-2/3*np.pi, -2/3*np.pi) # Target Orientation not rotated: [ 3.06161700e-17  8.66025404e-01  5.00000000e-01 -5.30287619e-17] # Rotated: [ 5.30287619e-17 -5.00000000e-01  8.66025404e-01  3.06161700e-17]

        # Convert Roll-Pitch-Yaw to Quaternion [w, x, y, z] format
        r = R.from_euler("xyz", [roll, pitch, yaw], degrees=False)
        quat_wxyz = r.as_quat(scalar_first=True)

        rotated_quat = self.ROT_180_Z * R.from_quat(quat_wxyz, scalar_first=True)
        rotated_quat_wxyz = rotated_quat.as_quat(scalar_first=True)

        # print(quat_wxyz)
        # print(rotated_quat_wxyz)

        # Apply 180-degree rotation to the position
        rotated_pos = self.ROT_180_Z.apply([pos_x, pos_y, pos_z])

        self.target_pose = np.concatenate((rotated_pos, rotated_quat_wxyz), axis=0)


    def get_predefined_pose(self):
        """Get a target pose from a predefined array of poses"""
        predefined_poses = np.array([
            [0.2, -0.4, 0.3, 0.0, 1.0, 0.0, 0.0],
            [0.15, -0.3, 0.15, 0.0, 1.0, 0.0, 0.0],
            [-0.1,  -0.5,  0.4, 0.0, 1.0, 0.0, 0.0],
        ])
        target_pose = predefined_poses[self.current_index]

        pos = target_pose[:3]  # [X, Y, Z]
        quat_wxyz = target_pose[3:]  # Quat in [W X Y Z]

        rotated_quat = self.ROT_180_Z * R.from_quat(quat_wxyz, scalar_first=True)
        rotated_quat_wxyz = rotated_quat.as_quat(scalar_first=True)

        # Apply 180-degree rotation to the position
        rotated_pos = self.ROT_180_Z.apply(pos)

        self.target_pose = np.concatenate((rotated_pos, rotated_quat_wxyz), axis=0)


    def axis_angle_to_quaternion(self, axis_angle):
        """
        Convert a single axis-angle representation to a quaternion (w, x, y, z),
        ensuring a consistent sign convention to prevent flipping.
        """
        # Convert axis-angle to quaternion
        quat_wxyz = R.from_rotvec(axis_angle).as_quat(scalar_first=True)#[0]  # Output format: [w, x, y, z]
        
        # Ensure quaternion continuity (prevent sign flips based on vector part)
        if self.prev_quaternion is not None:
            if np.dot(quat_wxyz[1:], self.prev_quaternion[1:]) < 0:  # Use only vector part [x, y, z]
                quat_wxyz *= -1  # Flip quaternion to maintain consistency
        # else:
        #     quat_wxyz *= -1

        self.prev_quaternion = quat_wxyz.copy()  # Store for next iteration

        return quat_wxyz


    def get_actual_tcp_pose(self):
        """
        Returns the actual TCP received from the robot in [X, Y, Z] + Quaternion in [w, x, y, z]
        and applies a 180-degree Z-axis rotation.
        """
        tcp_pose = np.array(self.rtde_r.getActualTCPPose())  # [x, y, z, rx, ry, rz]
        pos = tcp_pose[:3]
        axis_angle = tcp_pose[3:]

        # Convert Axis-Angle to Quaternion with proper formatting
        quat_wxyz = self.axis_angle_to_quaternion(axis_angle)  # Extract single quaternion

        # Apply 180-degree rotation
        rotated_quat = self.ROT_180_Z * R.from_quat(quat_wxyz, scalar_first=True) # Mark
        rotated_quat_wxyz = rotated_quat.as_quat(scalar_first=True)
        rotated_pos = self.ROT_180_Z.apply(pos)

        self.tcp_pose = np.concatenate((rotated_pos, rotated_quat_wxyz), axis=0)


    def enforce_axis_angle_continuity(self, new_rotvec, prev_rotvec):
        """Ensure smooth transitions in axis-angle representation by preventing sudden sign flips."""
        if prev_rotvec is None:
            return new_rotvec  # No previous value, nothing to compare against

        # Check overall flip using dot product
        if np.dot(new_rotvec, prev_rotvec) < 0:  
            new_rotvec *= -1  # Flip the entire vector

        return new_rotvec


    def execute_action_on_real_robot(self):
        """Send a TCP displacement action to the robot, ensuring it is rotated back 180 degrees before execution."""
        new_tcp = np.zeros(6)  
        new_tcp[:3] = self.tcp_pose[:3] + self.action[:3]  

        # Compute new rotation in quaternion space
        displacement_rotation = R.from_rotvec(self.action[3:])
        new_rotation = displacement_rotation * R.from_quat(self.tcp_pose[3:], scalar_first=True) # Mark #, scalar_first=True
        new_rotation_rotvec = new_rotation.as_rotvec()

        # Ensure axis-angle continuity (new robust method)
        new_rotation_rotvec = self.enforce_axis_angle_continuity(new_rotation_rotvec, self.prev_rotvec_2)
        self.prev_rotvec_2 = new_rotation_rotvec.copy()

        new_rotation = R.from_rotvec(new_rotation_rotvec)

        # Apply 180-degree rotation before converting to axis-angle
        final_rotation = self.ROT_180_Z.inv() * new_rotation  # Rotate quaternion back

        # Convert to axis-angle only at the last step
        new_rotvec = final_rotation.as_rotvec()

        # Ensure axis-angle continuity (new robust method)
        new_rotvec = self.enforce_axis_angle_continuity(new_rotvec, self.prev_rotvec)
        self.prev_rotvec = new_rotvec.copy()

        new_tcp[3:] = new_rotvec  # Store final axis-angle rotation
        new_tcp[:3] = self.ROT_180_Z.inv().apply(new_tcp[:3])  # Rotate position back

        print("Final TCP Pose Sent to Robot:", new_tcp)
        self.rtde_c.moveL(new_tcp.tolist(), speed=1.5, acceleration=1.5)
        # self.rtde_c.moveL(new_tcp.tolist(), speed=0.2, acceleration=0.2)


    # Function to get the robot's current state with quaternion in [w, x, y, z] format
    def get_robot_state(self):
        return np.concatenate((self.tcp_pose, self.target_pose, self.previous_action), axis=0)


    def quaternion_geodesic_distance(self, q1, q2):
        """Compute geodesic distance between two quaternions."""
        dot_product = np.clip(np.dot(q1, q2), -1.0, 1.0)  # Ensure valid range
        return 2 * np.arccos(abs(dot_product))  # Compute angle difference


    def run_on_real_robot(self):
        """Main control loop to move the robot towards sampled poses."""
        self.move_robot_to_home()
        
        if self.mode == "Predefined":
            self.get_predefined_pose()
        elif self.mode == "Sample":
            self.sample_random_pose()
        
        timestep = 0  

        while True:
            self.get_actual_tcp_pose()
            obs = self.get_robot_state()

            print("Current TCP Pose: ", self.tcp_pose)
            print("Target Pose: ", self.target_pose)

            with torch.inference_mode():
                self.action, _ = self.agent.predict(obs, deterministic=True)
                self.action *= self.action_scaling 
                # print(self.action)

            if self.save:
                self.save_observations_to_csv(timestep)

            self.previous_action = self.action
            self.execute_action_on_real_robot()  

            position_distance = np.linalg.norm(self.target_pose[:3] - self.tcp_pose[:3])

            current_quat = self.tcp_pose[3:]  
            target_quat = self.target_pose[3:]  
            orientation_distance = self.quaternion_geodesic_distance(current_quat, target_quat)

            print(f"Position Error: {position_distance}, Orientation Error: {orientation_distance}")

            if (position_distance < 0.01 and orientation_distance < 0.05):# or (position_distance < 0.01 and orientation_distance < 0.08 and timestep > 100):
                print("Target reached!")
                print("Final TCP Pose: ", self.tcp_pose)
                # return
                if self.mode == "Predefined":
                    self.current_index += 1
                    if self.current_index == 3:
                        return
                    self.get_predefined_pose()
                elif self.mode == "Sample":
                    self.sample_random_pose()
                # timestep = -1 # Reset timestep counter

            timestep += 1  


if __name__ == "__main__":
    robot_ip = "10.126.51.99"
    # robot_ip = "192.168.1.100"
    # model_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_stiffness_10000000_v2/su6rb80q/model.zip"
    # model_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_stiffness_10000000_without_gripper/yiv2hxjt/model.zip"
    model_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_stiffness_10000000_without_gripper_v2/87m5wbvy/model.zip"
    save_dir = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/"
    filename = "ursim_predefined_pose_rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_stiffness_10000000_without_gripper_0_0025_predefined_poses.csv"
    # filename = "real_observation_predefined_pose_rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_stiffness_10000000_without_gripper_0_01_random_poses.csv"
    action_scaling = 0.0025
    save = False # False # True
    mode = "Predefined" # Options available: "Sample" (sample uniformly from specified range), "Predefined" (predefined poses)

    controller = UR5eRobotController(robot_ip, model_path, action_scaling, save_dir, filename, mode, save)

    try:
        controller.run_on_real_robot()
    except KeyboardInterrupt:
        print("Stopping robot execution.")
    finally:
        controller.rtde_c.stopScript()
