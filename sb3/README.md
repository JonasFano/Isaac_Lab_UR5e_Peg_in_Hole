Contains all necessary information on how to start training and how to run a trained agent with the sb3 library.



####################################################
# IK Relative Control without domain randomization #
####################################################

## Peg in Hole Task (Cranfield Benchmark)
### Train PPO agent
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/train_sb3_ppo.py --num_envs 4 --task UR5e-Peg-in-hole-IK --headless


## Peg Insert Task (From Isaac Gym Factory)
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/train_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-IK --headless

    source isaaclab/bin/activate
    cd isaaclab/IsaacLab

    # rel_ik_sb3_ppo_ur5e_peg_insert_0_001_v9
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-IK --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/zfui520n/model.zip

    # rel_ik_sb3_ppo_ur5e_peg_insert_0_001_v10
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-IK --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/wzyxo5qn/model.zip

#################################################
# IK Relative Control with domain randomization #
#################################################


## Peg in Hole Task (Cranfield Benchmark)
### Train PPO agent
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Lift_Cube/sb3/train_sb3_ppo.py --num_envs 64 --task UR5e-Peg-in-hole-IK-Domain-Rand --headless



######################
# Weights and Biases #
######################

## Peg Insert Task
    source isaaclab/bin/activate
    cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3
    wandb sweep --project rel_ik_sb3_ppo_ur5e_peg_insert_0_001_v9 config_sb3_ppo.yaml # With force torque reward: 1.0 1.0 5.0 10.0 # Clamping force/torque +/- 5000 
    wandb sweep --project rel_ik_sb3_ppo_ur5e_peg_insert_0_001_v10 config_sb3_ppo.yaml # With force torque reward: 10.0 10.0 5.0 10.0 # Clamping force/torque +/- 5000 
    wandb sweep --project rel_ik_sb3_ppo_ur5e_peg_insert_0_001_v11 config_sb3_ppo.yaml # With force torque reward: 10.0 10.0 15.0 30.0 # Clamping force/torque +/- 5000 
    wandb sweep --project rel_ik_sb3_ppo_ur5e_peg_insert_0_001_v12 config_sb3_ppo.yaml # With force torque reward: 5.0 5.0 15.0 30.0 # Clamping force/torque +/- 5000 # From here curriculum-based reward shaping
    wandb sweep --project rel_ik_sb3_ppo_ur5e_peg_insert_0_001_v13 config_sb3_ppo.yaml # With force torque reward: 10.0 10.0 50.0 100.0 # Clamping force/torque +/- 5000 
    wandb sweep --project rel_ik_sb3_ppo_ur5e_peg_insert_0_001_v14 config_sb3_ppo.yaml # With force torque reward: 5.0 5.0 25.0 50.0 # Clamping force/torque +/- 5000 
    wandb sweep --project rel_ik_sb3_ppo_ur5e_peg_insert_0_001_v15 config_sb3_ppo.yaml # With force torque reward: 5.0 5.0 25.0 50.0 # Clamping force/torque +/- 5000 # Exact hole estimate
    wandb sweep --project rel_ik_sb3_ppo_ur5e_peg_insert_0_001_v16 config_sb3_ppo.yaml # With force torque reward: 5.0 5.0 25.0 50.0 # Clamping force/torque +/- 5000 # Exact hole estimate # Change position iteration count from 192 to 4
    wandb sweep --project rel_ik_sb3_ppo_ur5e_peg_insert_0_001_v17 config_sb3_ppo.yaml # With force torque reward: 5.0 5.0 25.0 50.0 # Clamping force/torque +/- 5000 # Exact hole estimate # Change position iteration count back to 192 
    wandb sweep --project rel_ik_sb3_ppo_ur5e_peg_insert_0_001_v18 config_sb3_ppo.yaml # With force torque reward: 5.0 5.0 25.0 50.0 # No clamping # Exact hole estimate # Less randomization