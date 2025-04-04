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
    wandb sweep --project rel_ik_sb3_ppo_ur5e_peg_insert_0_005 config_sb3_ppo.yaml
    wandb sweep --project rel_ik_sb3_ppo_ur5e_peg_insert_0_005_v2 config_sb3_ppo.yaml