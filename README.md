# Short project description
This project was part of my Master's Thesis with the title "Reinforcement Learning for Robot Control in Isaac Lab: A Feasibility Study" for the Master's program "Robot Systems - Advanced Robotics Technology" at the University of Southern Denmark (SDU). The task was to assess the feasibility of using RL-based robot control for a peg-in-hole task using the advanced physics simulator NVIDIA Isaac Lab. A stepwise development process was used in which task complexity is gradually increased to enable systematic optimization and validation of key framework components and algorithm hyperparameters. Each task builds directly on the previous one, reusing components and introducing new challenges in isolation.

This Repository includes the implementation to train PPO agents (from Stable-Baselines3) in Isaac Lab. The considered task includes a UR5e robot and requires the policy to insert an already grasped peg into a hole. The implemented controllers are relative differential inverse kinematics (IK) control, and, most importantly, impedance control (position tracking only).

This complex, contact-rich peg-in-hole task represents the final and most industrially relevant task of the thesis project. Due to GPU errors using the UR5e model and differential IK control, a custom Cartesian impedance controller was implemented. With that, a comparison of domain randomization strategies was performed.

The Weights&Biases tool was utilized to automate the hyperparameter search since it allows to extensively visualize the episode reward mean across training runs conducted with different hyperparameter configurations or task setups.



# Requirements
Follow these steps to create a virtual python environment and to install Isaac Sim and Isaac Lab (4.5.0):

https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

Install requirements:
    
    pip install -r /path/to/requirements.txt 


# Example video
## Unsuccessful insertions (UR5e + Diff. IK) 
https://youtu.be/k9X7QNSGh94

## Unsuccessful insertions (Franka + Diff. IK)
https://youtu.be/kdAlnAUcgPE

## Successful insertions 
https://youtu.be/mFXXH20iyYc



# Hyperparameter optimization with Weights&Biases
## PPO
### UR5e with impedance control

    source /path/to/virtual/environment/bin/activate
    cd /path/to/repository/sb3
    wandb sweep --project impedance_ctrl_peg_insert config_sb3_ppo.yaml 

Notably, to change the environment to domain randomization, the specific environment that is used for training has to be changed inside train_sb3_wandb_ppo.py. The task options are listed below. Hyperparameters and parameter sweep names have to be set inside config_sb3_ppo.yaml.



# Train PPO agent without Weights&Biases
Option 1:

    source /path/to/virtual/environment/bin/activate
    cd /path/to/repository/sb3
    python3 train_sb3_ppo.py --num_envs 2048 --task UR5e-Peg-Insert-Impedance-Ctrl --headless

Option 2:

    source /path/to/virtual/environment/bin/activate
    cd /path/to/isaac/lab/installation/directory
    ./isaaclab.sh -p /path/to/repository/sb3/train_sb3_ppo.py --num_envs 2048 --task UR5e-Peg-Insert-Impedance-Ctrl --headless

Tensorboard can be used to visualize training results

    tensorboard --logdir='directory'

Note: For this option, the hyperparameters are defined in /gym_env/env/agents/



# Examples to play PPO trained agent (MedPen-HoleNoise Policy)

    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-Impedance-Ctrl --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/ur5e/impedance_ctrl/impedance_ctrl_peg_insert_2048_envs_v14/ckohomxv/model.zip




# Task options (defined in /gym_env/env/__init__.py)
UR5e with Robotiq Hand-E Gripper, Relative Differential Inverse Kinematics Action Space and Domain Randomization

    --task UR5e-Peg-Insert-Impedance-Ctrl


Franka with Gripper, Relative Differential Inverse Kinematics Action Space and Domain Randomization

    --task Franka-Peg-Insert-Impedance-Ctrl


UR5e with Robotiq Hand-E Gripper, Cartesian Impedance Control and Domain Randomization

    --task UR5e-Peg-Insert-IK


Franka with Gripper, Cartesian Impedance Control and Domain Randomization

    --task Franka-Peg-Insert-IK


# Real-world execution
All trained models for the UR5e with impedance control are compatible with real-world execution.


## Instructions
    
1. Create physical environment with peg and hole (and grasp peg with the gripper).
2. Connect to the UR5e robot via Ethernet and set correct IP address.
3. Run `cartesian_impedance.script` on the teach pendant.
4. Run the python script using the terminal commands described below.

    source /path/to/virtual/environment/bin/activate
    cd /path/to/repository/real_world_execution
    python3 rtde_rl_impedance_control.py


> **Note:** An Isaac Lab installation is **not** required for real-world execution.



# Visualization scripts
Inside the utils-folder, several scripts are provided for analyzing and visualizing the data recorded during real-world deployment (rtde_rl_impedance_control.py) or simulation experiments (play_sb3_ppo_save_observations.py). The CSV files provided in the data-folder were utilized for the Master's Thesis and include both real-world experiment data and simulation test trials.