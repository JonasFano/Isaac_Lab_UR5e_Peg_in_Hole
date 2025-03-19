Contains all necessary information on how to start training and how to run a trained agent with the rl_games library.


#######################
# IK Relative Control #
#######################

## UR5e
### Train PPO agent
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/train_sb3_ppo.py --num_envs 4096 --task UR5e-Lift-Cube-IK --headless

For testing:

    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/train_sb3_ppo.py --num_envs 1 --task UR5e-Lift-Cube-IK --no_logging

Train pre-trained model:

    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/train_sb3_ppo.py --num_envs 256 --task UR5e-Lift-Cube-IK --checkpoint path/to/checkpoint


### Play the trained PPO agent
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --task UR5e-Lift-Cube-IK --num_envs 4 --checkpoint path/to/checkpoint


#### Examples:
Training runs for optimizing reward function (penalizing wrong TCP orientation) - Requires UR5e SDU gripper and object of scale=(0.3, 0.3, 1.0)

Reward weight too high

    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --task UR5e-Lift-Cube-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/logs/sb3/ppo/UR5e-Lift-Cube-IK/adjusted_reward_tcp_10/model.zip


    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --task UR5e-Lift-Cube-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/logs/sb3/ppo/UR5e-Lift-Cube-IK/adjusted_reward_tcp_7/model.zip

Optimal reward weight

    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --task UR5e-Lift-Cube-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/logs/sb3/ppo/UR5e-Lift-Cube-IK/adjusted_reward_tcp_6/model.zip
    
Reward weight too low

    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --task UR5e-Lift-Cube-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/logs/sb3/ppo/UR5e-Lift-Cube-IK/adjusted_reward_tcp_3/model.zip


Final training runs with SDU gripper

Min_height: 0.15 - Actuator stiffness: 1000.0 - Cube scale: (0.3, 0.3, 1.0) - Pose generation ranges: pos_x=(0.25, 0.35), pos_y=(0.3, 0.4), pos_z=(0.25, 0.35) - Unoise: 0.0 - Object reset range: "x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0) - Resampling time range: 5.0 - Episode length: 5.0 - Gripper offset: 0.135 - Robot reset: "position_range": (1.0, 1.0)

    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --task UR5e-Lift-Cube-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/models/rel_ik_sb3_ppo_ur5e_lift_cube_0_05_final_v2/rks36vpv/model.zip # Great performance - Reward 108 - num_envs 2048 - n_step 64
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --task UR5e-Lift-Cube-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/models/rel_ik_sb3_ppo_ur5e_lift_cube_0_05_final_v2/e5a16qaq/model.zip # Poor orientation alginment - Reward 94 -  num_envs 4096 - n_step 64

    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --task UR5e-Lift-Cube-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/models/rel_ik_sb3_ppo_ur5e_lift_cube_0_05_final_v3/bes84smk/model.zip # Great performance - Reward 118 - num_envs 4096 - n_step 64 - 8 hours
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --task UR5e-Lift-Cube-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/models/rel_ik_sb3_ppo_ur5e_lift_cube_0_05_final_v3/4pssftbx/model.zip # Best performance - Reward 126 - num_envs 4096 - n_step 128 - 8 hours

Final training runs with Hand E gripper: rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_final

Min_height: 0.08 - Actuator stiffness: 1000.0 - Cube scale: (0.4, 0.4, 0.4) - Pose generation ranges: pos_x=(0.25, 0.35), pos_y=(0.3, 0.4), pos_z=(0.25, 0.35) - Unoise: 0.0 - Object reset range: "x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0) - Resampling time range: 5.0 - Episode length: 5.0 - Gripper offset: 0.15 - Robot reset: "position_range": (1.0, 1.0)

    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --task UR5e-Hand-E-Lift-Cube-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/models/rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_final/hr7kc81h/model.zip # Great performance - Reward 117 - num_envs 4096 - n_step 32
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --task UR5e-Hand-E-Lift-Cube-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/models/rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_final/5jg855iw/model.zip # Great performance - Reward 122 - num_envs 4096 - n_step 32
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --task UR5e-Hand-E-Lift-Cube-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/models/rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_final/l5zzjm5z/model.zip # Great performance - Reward 119 - num_envs 4096 - n_step 64



## Franka
### Train PPO agent
source isaaclab/bin/activate
cd isaaclab/IsaacLab
./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/train_sb3_ppo.py --num_envs 4096 --task Franka-Lift-Cube-IK --headless


### Play the trained PPO agent
./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --task Franka-Lift-Cube-IK --num_envs 4 --checkpoint path/to/checkpoint



#################################################
# IK Relative Control with domain randomization #
#################################################

## UR5e
### Train PPO agent
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/train_sb3_ppo.py --num_envs 64 --task UR5e-Hand-E-Domain-Rand-Lift-Cube-IK --headless


# rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_without_friction: Actuator stiffness: 1000 (3000 for gripper) - Cube Scale: (0.4, 0.4, 0.4) - Object Pose Generation: pos_x=(0.25, 0.35), pos_y=(0.3, 0.4), pos_z=(0.25, 0.35), roll=0, pitch=pi, yaw=(-pi, +pi) - Action scaling: 0.05 - Gripper binary position action: Open: 0.0, Close: -0.025 - Unoise: 0.0 - TCP offset: [0.0, 0.0, 0.15] - Robot Reset: "position_range" (0.9, 1.1) - Reset Object Position: "x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0) - No mass randomization (mass = 0.5) - Robot Actuator randomization: "stiffness_distribution_params": (0.7, 1.3), "damping_distribution_params": (0.7, 1.3) - Gripper randomization: "stiffness_distribution_params": (2500, 3500), "damping_distribution_params": (300, 700) - No Gripper Friction randomization - No Object Friction randomization - Reaching object: std: 0.1, weight: 1.0 - Lifting object: minheight: 0.08, weight: 25.0 - Object goal tracking: std: 0.3, weight: 16.0 - Object goal tracking: std: 0.05, weight: 5.0 - Orientation tracking: weight: -6.0 - Action rate: weight: -1e-4 - Object dropping: min height: -0.05 - Action rate curriculum: weight: -1e-1, num_steps: 10000 - Decimation 2 - Dt 0.01 - Episode length: 5.0
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --num_envs 4 --task UR5e-Hand-E-Domain-Rand-Lift-Cube-IK --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/models/rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_without_friction/oaih8t82/model.zip



# rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass: Actuator stiffness: 1000 (3000 for gripper) - Cube Scale: (0.4, 0.4, 0.4) - Object Pose Generation: pos_x=(0.25, 0.35), pos_y=(0.3, 0.4), pos_z=(0.25, 0.35), roll=0, pitch=pi, yaw=(-pi, +pi) - Action scaling: 0.05 - Gripper binary position action: Open: 0.0, Close: -0.025 - Unoise: 0.0 - TCP offset: [0.0, 0.0, 0.15] - Robot Reset: "position_range" (0.9, 1.1) - Reset Object Position: "x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0) - No mass randomization (mass = 0.5) - Robot Actuator randomization: "stiffness_distribution_params": (0.7, 1.3), "damping_distribution_params": (0.7, 1.3) - Gripper randomization: "stiffness_distribution_params": (2500, 3500), "damping_distribution_params": (300, 700) - Gripper Friction randomization: "stiffness_distribution_params": (0.85, 0.9), "damping_distribution_params": (0.6, 0.7), "restitution_distribution_params": (0.2, 0.6) - Object Friction randomization: "stiffness_distribution_params": (0.7, 0.8), "damping_distribution_params": (0.5, 0.6), "restitution_distribution_params": (0.3, 0.7) - Reaching object: std: 0.1, weight: 1.0 - Lifting object: minheight: 0.08, weight: 25.0 - Object goal tracking: std: 0.3, weight: 16.0 - Object goal tracking: std: 0.05, weight: 5.0 - Orientation tracking: weight: -6.0 - Action rate: weight: -1e-4 - Object dropping: min height: -0.05 - Action rate curriculum: weight: -1e-1, num_steps: 10000 - Decimation 2 - Dt 0.01 - Episode length: 5.0
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --num_envs 4 --task UR5e-Hand-E-Domain-Rand-Lift-Cube-IK --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/models/rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass/unk8h7dj/model.zip


# rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v2: Actuator stiffness: 1000 (3000 for gripper) - Cube Scale: (0.4, 0.4, 0.4) - Object Pose Generation: pos_x=(0.25, 0.35), pos_y=(0.3, 0.4), pos_z=(0.25, 0.35), roll=0, pitch=pi, yaw=(-pi, +pi) - Action scaling: 0.05 - Gripper binary position action: Open: 0.0, Close: -0.025 - Unoise: 0.0 - TCP offset: [0.0, 0.0, 0.15] - Robot Reset: "position_range" (1.0, 1.0) - Reset Object Position: "x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0) - No mass randomization (mass = 0.5) - Robot Actuator randomization: "stiffness_distribution_params": (0.7, 1.3), "damping_distribution_params": (0.7, 1.3) - Gripper randomization: "stiffness_distribution_params": (2500, 3500), "damping_distribution_params": (300, 700) - Gripper Friction randomization: "stiffness_distribution_params": (0.85, 0.9), "damping_distribution_params": (0.6, 0.7), "restitution_distribution_params": (0.1, 0.1) - Object Friction randomization: "stiffness_distribution_params": (0.7, 0.8), "damping_distribution_params": (0.5, 0.6), "restitution_distribution_params": (0.1, 0.1) - Reaching object: std: 0.1, weight: 1.0 - Lifting object: minheight: 0.08, weight: 35.0 - Object goal tracking: std: 0.3, weight: 16.0 - Object goal tracking fine grained: std: 0.05, weight: 5.0 - Orientation tracking: weight: -6.0 - Action rate: weight: -1e-4 - Object dropping: min height: -0.05 - Action rate curriculum: weight: -1e-1, num_steps: 100000 - Decimation 2 - Dt 0.01 - Episode length: 5.0
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --num_envs 4 --task UR5e-Hand-E-Domain-Rand-Lift-Cube-IK --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/models/rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v2/cp6jy1um/model.zip



# rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v3: Actuator stiffness: 1000 (3000 for gripper) - Cube Scale: (0.4, 0.4, 0.4) - Object Pose Generation: pos_x=(0.25, 0.35), pos_y=(0.3, 0.4), pos_z=(0.25, 0.35), roll=0, pitch=pi, yaw=(-pi, +pi) - Action scaling: 0.05 - Gripper binary position action: Open: 0.0, Close: -0.025 - Unoise: 0.0 - TCP offset: [0.0, 0.0, 0.15] - Robot Reset: "position_range" (1.0, 1.0) - Reset Object Position: "x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0) - No mass randomization (mass = 0.5) - Robot Actuator randomization: "stiffness_distribution_params": (0.7, 1.3), "damping_distribution_params": (0.7, 1.3) - Gripper randomization: "stiffness_distribution_params": (2500, 3500), "damping_distribution_params": (300, 700) - Gripper Friction randomization: "stiffness_distribution_params": (1.5, 1.5), "damping_distribution_params": (1.5, 1.5), "restitution_distribution_params": (0.1, 0.1) - Object Friction randomization: "stiffness_distribution_params": (1.5, 1.5), "damping_distribution_params": (1.5, 1.5), "restitution_distribution_params": (0.1, 0.1) - Reaching object: std: 0.1, weight: 1.0 - Lifting object: minheight: 0.08, weight: 35.0 - Object goal tracking: std: 0.3, weight: 16.0 - Object goal tracking: std: 0.05, weight: 5.0 - Orientation tracking: weight: -6.0 - Action rate: weight: -1e-4 - Object dropping: min height: -0.05 - Action rate curriculum: weight: -1e-1, num_steps: 100000 - Decimation 2 - Dt 0.01 - Episode length: 5.0
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --num_envs 4 --task UR5e-Hand-E-Domain-Rand-Lift-Cube-IK --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/models/rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v3/qeumhda6/model.zip



# rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v4: Actuator stiffness: 1000 (3000 for gripper) - Cube Scale: (0.4, 0.4, 0.4) - Object Pose Generation: pos_x=(0.25, 0.35), pos_y=(0.3, 0.4), pos_z=(0.25, 0.35), roll=0, pitch=pi, yaw=(-pi, +pi) - Action scaling: 0.05 - Gripper binary position action: Open: 0.0, Close: -0.025 - Unoise: 0.0 - TCP offset: [0.0, 0.0, 0.15] - Robot Reset: "position_range" (1.0, 1.0) - Reset Object Position: "x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0) - No mass randomization (mass = 0.5) - Robot Actuator randomization: "stiffness_distribution_params": (0.7, 1.3), "damping_distribution_params": (0.7, 1.3) - Gripper randomization: "stiffness_distribution_params": (2500, 3500), "damping_distribution_params": (300, 700) - Gripper Friction randomization: "stiffness_distribution_params": (1.2, 1.2), "damping_distribution_params": (1.2, 1.2), "restitution_distribution_params": (0.1, 0.1) - Object Friction randomization: "stiffness_distribution_params": (1.2, 1.2), "damping_distribution_params": (1.2, 1.2), "restitution_distribution_params": (0.1, 0.1) - Reaching object: std: 0.1, weight: 1.0 - Lifting object: minheight: 0.08, weight: 35.0 - Object goal tracking: std: 0.3, weight: 16.0 - Object goal tracking: std: 0.05, weight: 5.0 - Orientation tracking: weight: -6.0 - Action rate: weight: -1e-4 - Object dropping: min height: -0.05 - Action rate curriculum: weight: -1e-1, num_steps: 100000 - Decimation 2 - Dt 0.01 - Episode length: 5.0
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --num_envs 4 --task UR5e-Hand-E-Domain-Rand-Lift-Cube-IK --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/models/rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v4/l5jtb0b3/model.zip



# rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v5: Actuator stiffness: 1000 (3000 for gripper) - Cube Scale: (0.4, 0.4, 0.4) - Object Pose Generation: pos_x=(0.25, 0.35), pos_y=(0.3, 0.4), pos_z=(0.25, 0.35), roll=0, pitch=pi, yaw=(-pi, +pi) - Action scaling: 0.05 - Gripper binary position action: Open: 0.0, Close: -0.025 - Unoise: 0.0 - TCP offset: [0.0, 0.0, 0.15] - Robot Reset: "position_range" (1.0, 1.0) - Reset Object Position: "x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0) - No mass randomization (mass = 0.5) - Robot Actuator randomization: "stiffness_distribution_params": (0.7, 1.3), "damping_distribution_params": (0.7, 1.3) - Gripper randomization: "stiffness_distribution_params": (2500, 3500), "damping_distribution_params": (300, 700) - Gripper Friction randomization: "stiffness_distribution_params": (1.2, 1.2), "damping_distribution_params": (1.2, 1.2), "restitution_distribution_params": (0.1, 0.1) - Object Friction randomization: "stiffness_distribution_params": (0.8, 0.8), "damping_distribution_params": (0.8, 0.8), "restitution_distribution_params": (0.1, 0.1) - Reaching object: std: 0.1, weight: 1.0 - Lifting object: minheight: 0.08, weight: 35.0 - Object goal tracking: std: 0.3, weight: 16.0 - Object goal tracking: std: 0.05, weight: 5.0 - Orientation tracking: weight: -6.0 - Action rate: weight: -1e-4 - Object dropping: min height: -0.05 - Action rate curriculum: weight: -1e-1, num_steps: 100000 - Decimation 2 - Dt 0.01 - Episode length: 5.0
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --num_envs 4 --task UR5e-Hand-E-Domain-Rand-Lift-Cube-IK --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/models/rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v5/jrmj0wxl/model.zip



# rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v6: Actuator stiffness: 1000 (3000 for gripper) - Cube Scale: (0.4, 0.4, 0.4) - Object Pose Generation: pos_x=(0.25, 0.35), pos_y=(0.3, 0.4), pos_z=(0.25, 0.35), roll=0, pitch=pi, yaw=(-pi, +pi) - Action scaling: 0.05 - Gripper binary position action: Open: 0.0, Close: -0.025 - Unoise: 0.0 - TCP offset: [0.0, 0.0, 0.15] - Robot Reset: "position_range" (1.0, 1.0) - Reset Object Position: "x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0) - No mass randomization (mass = 0.5) - Robot Actuator randomization: "stiffness_distribution_params": (0.7, 1.3), "damping_distribution_params": (0.7, 1.3) - Gripper randomization: "stiffness_distribution_params": (2500, 3500), "damping_distribution_params": (300, 700) - Gripper Friction randomization: "stiffness_distribution_params": (1.0, 1.0), "damping_distribution_params": (0.8, 0.8), "restitution_distribution_params": (0.1, 0.1) - Object Friction randomization: "stiffness_distribution_params": (0.8, 0.8), "damping_distribution_params": (0.6, 0.6), "restitution_distribution_params": (0.1, 0.1) - Reaching object: std: 0.1, weight: 1.0 - Lifting object: minheight: 0.08, weight: 35.0 - Object goal tracking: std: 0.3, weight: 16.0 - Object goal tracking: std: 0.05, weight: 5.0 - Orientation tracking: weight: -6.0 - Action rate: weight: -1e-4 - Object dropping: min height: -0.05 - Action rate curriculum: weight: -1e-1, num_steps: 100000 - Decimation 2 - Dt 0.01 - Episode length: 5.0
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --num_envs 4 --task UR5e-Hand-E-Domain-Rand-Lift-Cube-IK --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/models/rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v6/pe6uzbo5/model.zip



# rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v7: Actuator stiffness: 1000 (3000 for gripper) - Cube Scale: (0.4, 0.4, 0.4) - Object Pose Generation: pos_x=(0.25, 0.35), pos_y=(0.3, 0.4), pos_z=(0.25, 0.35), roll=0, pitch=pi, yaw=(-pi, +pi) - Action scaling: 0.05 - Gripper binary position action: Open: 0.0, Close: -0.025 - Unoise: 0.0 - TCP offset: [0.0, 0.0, 0.15] - Robot Reset: "position_range" (1.0, 1.0) - Reset Object Position: "x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0) - No mass randomization (mass = 0.5) - Robot Actuator randomization: "stiffness_distribution_params": (0.7, 1.3), "damping_distribution_params": (0.7, 1.3) - Gripper randomization: "stiffness_distribution_params": (2500, 3500), "damping_distribution_params": (300, 700) - Gripper Friction randomization: "stiffness_distribution_params": (1.0, 1.0), "damping_distribution_params": (0.8, 0.8), "restitution_distribution_params": (0.1, 0.1) - Object Friction randomization: "stiffness_distribution_params": (0.8, 0.8), "damping_distribution_params": (0.6, 0.6), "restitution_distribution_params": (0.1, 0.1) - Reaching object: std: 0.1, weight: 1.0 - Lifting object: minheight: 0.05, weight: 35.0 - Object goal tracking: std: 0.3, weight: 16.0 - Object goal tracking: std: 0.05, weight: 5.0 - Orientation tracking: weight: -6.0 - Action rate: weight: -1e-4 - Object dropping: min height: -0.05 - Action rate curriculum: weight: -1e-1, num_steps: 100000 - Decimation 2 - Dt 0.01 - Episode length: 5.0
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --num_envs 4 --task UR5e-Hand-E-Domain-Rand-Lift-Cube-IK --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/models/rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v7/xlbrr2gw/model.zip



# rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v8: Actuator stiffness: 10000000 (Gripper damping: 50000) - Cube Scale: (0.4, 0.4, 0.4) - Object Pose Generation: pos_x=(0.25, 0.35), pos_y=(0.3, 0.4), pos_z=(0.25, 0.35), roll=0, pitch=pi, yaw=(-pi, +pi) - Action scaling: 0.05 - Gripper binary position action: Open: 0.0, Close: -0.025 - Unoise: 0.0 - TCP offset: [0.0, 0.0, 0.15] - Robot Reset: "position_range" (1.0, 1.0) - Reset Object Position: "x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0) - No mass randomization (mass = 0.5) - Robot Actuator randomization: "stiffness_distribution_params": (0.7, 1.3), "damping_distribution_params": (0.7, 1.3) - Gripper randomization: "stiffness_distribution_params": (2500, 3500), "damping_distribution_params": (300, 700) - Gripper Friction randomization: "stiffness_distribution_params": (1.0, 1.0), "damping_distribution_params": (0.8, 0.8), "restitution_distribution_params": (0.1, 0.1) - Object Friction randomization: "stiffness_distribution_params": (0.8, 0.8), "damping_distribution_params": (0.6, 0.6), "restitution_distribution_params": (0.1, 0.1) - Reaching object: std: 0.1, weight: 1.0 - Lifting object: minheight: 0.05, weight: 35.0 - Object goal tracking: std: 0.3, weight: 16.0 - Object goal tracking: std: 0.05, weight: 5.0 - Orientation tracking: weight: -6.0 - Action rate: weight: -1e-4 - Object dropping: min height: -0.05 - Action rate curriculum: weight: -1e-1, num_steps: 100000 - Decimation 2 - Dt 0.01 - Episode length: 5.0
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/play_sb3_ppo.py --num_envs 4 --task UR5e-Hand-E-Domain-Rand-Lift-Cube-IK --checkpoint


#################################################
# IK Absolute Control with domain randomization #
#################################################

## UR5e
### Train PPO agent
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/train_sb3_ppo.py --num_envs 64 --task UR5e-Hand-E-Domain-Rand-Lift-Cube-Abs-IK --headless


### Weights and Biases
    source isaaclab/bin/activate
    cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3
    wandb sweep --project abs_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_without_friction_and_mass config_sb3_ppo_domain_rand.yaml



##########################
# Joint Position Control #
##########################

## UR5e
### Train PPO agent
source isaaclab/bin/activate
cd isaaclab/IsaacLab
./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/train_sb3_ppo.py --num_envs 512 --task UR5e-Lift-Cube --headless


### Play the trained PPO agent
./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/play_sb3_ppo.py --task UR5e-Lift-Cube --num_envs 4 --checkpoint path/to/checkpoint



###################################
# Wandb (Weights and Biases) UR5e #
###################################

## PPO
    source isaaclab/bin/activate
    cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3
    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_final config_0_05.yaml

    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e config_0_05.yaml

    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_num_env_test config_sb3_ppo.yaml

    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_final_v2 config_sb3_ppo.yaml

    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_final_v3 config_sb3_ppo.yaml

    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_final_v4 config_sb3_ppo.yaml

    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_final_v5 config_sb3_ppo.yaml

    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_final config_sb3_ppo.yaml




## PPO with domain randomization
    source isaaclab/bin/activate
    cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3

    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand config_sb3_ppo_domain_rand.yaml

    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_without_friction config_sb3_ppo_domain_rand.yaml

    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass config_sb3_ppo_domain_rand.yaml
    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v2 config_sb3_ppo_domain_rand.yaml
    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v3 config_sb3_ppo_domain_rand.yaml
    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v4 config_sb3_ppo_domain_rand.yaml
    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v5 config_sb3_ppo_domain_rand.yaml
    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v6 config_sb3_ppo_domain_rand.yaml
    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v7 config_sb3_ppo_domain_rand.yaml
    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v8 config_sb3_ppo_domain_rand.yaml
    wandb sweep --project rel_ik_sb3_ppo_ur5e_lift_cube_0_05_hand_e_domain_rand_with_small_friction_and_without_mass_v9 config_sb3_ppo_domain_rand.yaml


## DDPG
    source isaaclab/bin/activate
    cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3
    wandb sweep --project rel_ik_sb3_ddpg_ur5e_lift_cube_0_05 config_sb3_ddpg.yaml


## TD3
    source isaaclab/bin/activate
    cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3
    wandb sweep --project rel_ik_sb3_td3_ur5e_lift_cube_0_05 config_sb3_td3.yaml

    wandb sweep --project rel_ik_sb3_td3_ur5e_lift_cube_0_05_noise_1_0 config_sb3_td3.yaml
    
    wandb sweep --project rel_ik_sb3_td3_ur5e_lift_cube_0_05_noise_100 config_sb3_td3.yaml

    wandb sweep --project rel_ik_sb3_td3_ur5e_lift_cube_0_05_bayes_64 config_sb3_td3.yaml
