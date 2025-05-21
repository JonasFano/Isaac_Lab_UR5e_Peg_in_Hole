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
### Train
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/train_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-IK --headless


### Play
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab

    # rel_ik_sb3_ppo_ur5e_peg_insert_0_001_v9
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-IK --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/ur5e/zfui520n/model.zip

    # rel_ik_sb3_ppo_ur5e_peg_insert_0_001_v10
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-IK --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/ur5e/wzyxo5qn/model.zip



## Peg Insert Task using Franka Panda
### Train
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/train_sb3_ppo.py --num_envs 1 --task Franka-Peg-Insert-IK --headless


### Play
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task Franka-Peg-Insert-IK --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/franka/e3m611ym/model.zip

    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task Franka-Peg-Insert-IK --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/franka/3w6krpeh/model.zip




#####################
# Impedance Control #
#####################
## UR5e
### Train
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/train_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-Impedance-Ctrl

### Wandb
    wandb sweep --project impedance_ctrl_peg_insert config_sb3_ppo.yaml 
    
    # impedance_ctrl_peg_insert_128_envs # stiffness=[10, 10, 50, 20, 20, 20]
    # impedance_ctrl_peg_insert_512_envs # stiffness=[300, 300, 300, 100, 100, 100] damping_ratio = 4
    # impedance_ctrl_peg_insert_1024_envs # stiffness=[400, 400, 400, 50, 50, 50] damping_ratio = 4
    # impedance_ctrl_peg_insert_2048_envs # stiffness=[300, 300, 300, 1000, 1000, 1000] damping_ratio = 4 (damping ratio 8 for z position)
    # impedance_ctrl_peg_insert_2048_envs_v2 # stiffness=[400, 400, 400, 1000, 1000, 1000] damping_ratio = 6 (damping ratio 8 for z position)
    # impedance_ctrl_peg_insert_2048_envs_v3 # stiffness=[400, 400, 400, 750, 750, 750] damping_ratio = 6 (damping ratio 8 for z position)
    # impedance_ctrl_peg_insert_2048_envs_v4 # stiffness=[350, 350, 350, 900, 900, 900] damping_ratio = 6 (damping ratio 8 for z position) - with contact wrench penalty (force weight: -0.005, torque weight: -0.025)
    # impedance_ctrl_peg_insert_2048_envs_v5 # stiffness=[350, 350, 350, 900, 900, 900] damping_ratio = 6 (damping ratio 8 for z position) - with contact wrench penalty (force weight: -0.005, torque weight: -0.025)

### Play
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-Impedance-Ctrl --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/ur5e/impedance_ctrl/ms518sqi/model.zip # 128 envs - stiffness=[10, 10, 50, 20, 20, 20] - damping_ratio = 1
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-Impedance-Ctrl --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/ur5e/impedance_ctrl/r61ror63/model.zip # 128 envs - stiffness=[10, 10, 50, 20, 20, 20] - damping_ratio = 1

    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-Impedance-Ctrl --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/ur5e/impedance_ctrl/wmkv2euq/model.zip # 512 envs - stiffness=[300, 300, 300, 100, 100, 100] - damping_ratio = 4
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-Impedance-Ctrl --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/ur5e/impedance_ctrl/wmkv2euq/model.zip # 512 envs - stiffness=[400, 400, 400, 50, 50, 50] - damping_ratio = 4

    # impedance_ctrl_peg_insert_2048_envs # Action Scale: 0.01
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-Impedance-Ctrl --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/ur5e/impedance_ctrl/impedance_ctrl_peg_insert_2048_envs/cvdhh2cg/model.zip
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-Impedance-Ctrl --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/ur5e/impedance_ctrl/impedance_ctrl_peg_insert_2048_envs/r9x5uvm1/model.zip 

    # impedance_ctrl_peg_insert_2048_envs_v2 # Action Scale: 0.01
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-Impedance-Ctrl --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/ur5e/impedance_ctrl/impedance_ctrl_peg_insert_2048_envs_v2/hrure71c/model.zip
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-Impedance-Ctrl --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/ur5e/impedance_ctrl/impedance_ctrl_peg_insert_2048_envs_v2/s9io8zqc/model.zip

    # impedance_ctrl_peg_insert_2048_envs_v3 # Weights: 20.0 - 20.0 - 15.0 - 200.0 # Action Scale: 0.01
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-Impedance-Ctrl --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/ur5e/impedance_ctrl/impedance_ctrl_peg_insert_2048_envs_v3/ecwzweqi/model.zip

    # impedance_ctrl_peg_insert_2048_envs_v4  # Weights: 50.0 - 50.0 - 20.0 - 35.0 - 50.0 - 250.0 - -0.005 - -0.025 # Action Scale: 0.01


    # impedance_ctrl_peg_insert_2048_envs_v5  # Weights: 40.0 - 40.0 - 25.0 - 250.0 - -100 - -0.005 - -0.025 # Action Scale: 0.01
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-Impedance-Ctrl --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/9svff3q2/model.zip

    # impedance_ctrl_peg_insert_2048_envs_v6 # Weights: 20.0 - 20.0 - 50.0 - 200.0 - -50.0 - -0.01 - -0.05
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-Impedance-Ctrl --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/ur5e/impedance_ctrl/impedance_ctrl_peg_insert_2048_envs_v6/po5be5ca/model.zip

    # impedance_ctrl_peg_insert_2048_envs_v7 # Weights: 25.0 - 25.0 - 100.0 - 500.0 - -50.0 - -0.01 - -0.05 # Action Scale: 0.01
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-Impedance-Ctrl --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/ur5e/impedance_ctrl/impedance_ctrl_peg_insert_2048_envs_v7/ls6utfi9/model.zip

    # impedance_ctrl_peg_insert_2048_envs_v8 # Weights: 20.0 - 20.0 - 500.0 - 1000.0 - -50.0 - -0.005 - -0.025 # Action Scale: 0.005
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-Impedance-Ctrl --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/ur5e/impedance_ctrl/impedance_ctrl_peg_insert_2048_envs_v8/b9ifeoyf/model.zip

    # impedance_ctrl_peg_insert_2048_envs_v9 # Weights: 20.0 - 50.0 - 200.0 - 10000.0 - -50.0 - -0.005 - -0.025 # Action scale: 0.005 # Without reset if outside of hole
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-Impedance-Ctrl --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/ur5e/impedance_ctrl/impedance_ctrl_peg_insert_2048_envs_v9/oq7xgaas/model.zip
    
    # impedance_ctrl_peg_insert_2048_envs_v10 # Weights: 50.0 - 100.0 - 200.0 - 5000.0 - -250.0 - -0.005 - -0.025 # Without reset if outside of hole
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task UR5e-Peg-Insert-Impedance-Ctrl --checkpoint 

    # impedance_ctrl_peg_insert_2048_envs_v11 # Weights: 50.0 - 100.0 - 100.0 - 5000.0 - -50.0 - -0.005 - -0.025 # Added sequential_keypoint_distance reward



## Franka
### Train
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/train_sb3_ppo.py --num_envs 1 --task Franka-Peg-Insert-Impedance-Ctrl

### Play
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/play_sb3_ppo.py --num_envs 1 --task Franka-Peg-Insert-Impedance-Ctrl --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/sb3/models/ms518sqi/model.zip



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
    wandb sweep --project rel_ik_sb3_ppo_ur5e_peg_insert_0_001_v17 config_sb3_ppo.yaml # With force torque reward: 5.0 5.0 25.0 50.0 # No clamping # Exact hole estimate # Less randomization
    wandb sweep --project rel_ik_sb3_ppo_ur5e_peg_insert_0_001_v18 config_sb3_ppo.yaml # With force torque reward: 5.0 5.0 25.0 50.0 # No clamping # Exact hole estimate # Less randomization
    wandb sweep --project rel_ik_sb3_ppo_ur5e_peg_insert_0_001_v19 config_sb3_ppo.yaml # With force torque reward: 5.0 5.0 25.0 50.0 # No clamping # Exact hole estimate # Less randomization


## Franka
    wandb sweep --project rel_ik_sb3_ppo_franka_peg_insert config_sb3_ppo.yaml
