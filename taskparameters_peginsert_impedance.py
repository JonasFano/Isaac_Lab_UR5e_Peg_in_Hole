import math
from isaaclab.managers import SceneEntityCfg

class TaskParams:
    #################################
    ### General Simulation Params ###
    #################################
    decimation = 20 #2
    episode_length_s = 5.0 # 5.0 # 10.0  # 10.0 # 0.5 # 5.0
    dt = 1/1000 # 0.01
    render_interval = 5
    gravity = [0.0, 0.0, -9.81]


    #########################
    ### Impedance Control ###
    #########################
    command_type = "pose"
    use_relative_mode = True
    ik_method = "dls"
    action_scale= 0.005 # 0.005 # 0.0

    gravity_compensation = True
    coriolis_centrifugal_compensation = True
    inertial_dynamics_decoupling = True
    max_torque_clamping = None # Array of max torques to clamp computed torques - no clamping if None - [150.0, 150.0, 150.0, 28.0, 28.0, 28.0] for physical UR5e

    # # impedance_ctrl_peg_insert_512_envs
    # stiffness = [300, 300, 300, 100, 100, 100]
    # damping_ratio = 4
    # damping_ratio_z = 4

    # # impedance_ctrl_peg_insert_2048_envs
    # stiffness = [300, 300, 300, 1000, 1000, 1000]
    # damping_ratio = 4
    # damping_ratio_z = 8

    # impedance_ctrl_peg_insert_2048_envs_v2
    # stiffness = [400, 400, 400, 1000, 1000, 1000]
    # damping_ratio = 6
    # damping_ratio_z = 8

    # # impedance_ctrl_peg_insert_2048_envs_v3
    # stiffness = [400, 400, 400, 750, 750, 750]
    # damping_ratio = 6
    # damping_ratio_z = 8

    # # impedance_ctrl_peg_insert_2048_envs_v4 impedance_ctrl_peg_insert_2048_envs_v5 impedance_ctrl_peg_insert_2048_envs_v6
    # stiffness = [350, 350, 350, 900, 900, 900]
    # damping_ratio = 6
    # damping_ratio_z = 8

    # impedance_ctrl_peg_insert_2048_envs_v7 # impedance_ctrl_peg_insert_2048_envs_v8 # impedance_ctrl_peg_insert_2048_envs_v9 # impedance_ctrl_peg_insert_2048_envs_v10 # impedance_ctrl_peg_insert_2048_envs_v11
    # stiffness = [300, 300, 300, 900, 900, 900]
    # damping_ratio = 6
    # damping_ratio_z = 8

    # impedance_ctrl_peg_insert_2048_envs_v12 # impedance_ctrl_peg_insert_2048_envs_v13 # impedance_ctrl_peg_insert_2048_envs_v14 # impedance_ctrl_peg_insert_2048_envs_v15 # impedance_ctrl_peg_insert_2048_envs_v16
    stiffness = [300, 300, 300, 850, 850, 850]
    damping_ratio = 6
    damping_ratio_z = 8


    damping = None # None = Critically damped



    #######################
    # Gains Randomization #
    #######################
    stiffness_ranges = {
        "0": (100, 500),  # x
        "2": (100, 500),  # z
        "3": (700, 1200), # rx
    }

    damping_ratio_ranges = {
        "0": (2.0, 6.0),
        "2": (6.0, 10.0),
        "3": (2.0, 6.0),
    }

    groupings = [
        [0, 1],     # x = y
        [2],        # z
        [3, 4, 5],  # rx = ry = rz
    ]


    ################
    # Observations #
    ################
    tcp_pose_unoise_min = -0.0001 # 0.1 mm
    tcp_pose_unoise_max = 0.0001 # 0.1 mm

    noise_std_hole_pose = 0.0025 # 0.0025 # 2.5 mm


    #########
    # Event #
    #########
    ik_max_iters = 15
    pos_error_threshold = 1e-3
    angle_error_threshold = 1e-3
    levenberg_marquardt_lambda = 0.01


    ##############
    ### Reward ###
    ##############
    action_rate_weight = -1e-4
    action_rate_curriculum_weight = -1e-1
    action_rate_curriculum_num_steps = 50000

    # Keypoint distance
    num_keypoints = 4
    coarse_kernel_a = 50
    coarse_kernel_b = 2
    keypoint_distance_coarse_weight = 50.0

    fine_kernel_a = 100
    fine_kernel_b = 0
    keypoint_distance_fine_weight = 100.0

    # Is peg centered
    is_peg_centered_xy_threshold = 0.003 # 2.5 mm l2 norm
    is_peg_centered_z_threshold = 0.09 # 8 cm
    is_peg_centered_z_variability = 0.005 # 2 mm
    is_peg_centered_weight = 100.0

    # Is peg inserted
    is_peg_inserted_weight = 10000.0

    # Peg falls of hole edge
    peg_missed_hole_weight = -100

    episode_ends_weight = -50

    # is_peg_centered_z_variability_top = -0.015
    # is_peg_centered_z_variability_middle = -0.012
    # is_peg_centered_z_variability_bottom = -0.009
    # is_peg_centered_weight_top = 20.0
    # is_peg_centered_weight_middle = 35.0
    # is_peg_centered_weight_bottom = 50.0

    # Contact wrench penalty
    force_penalty_weight = -5.0
    torque_penalty_weight = -10.0
    
    ###################
    ### Termination ###
    ###################
    object_dropping_min_height = -0.05 

    termination_height = 0.07
    xy_margin = 0.0155 # half-size buffer beyond the hole's bounding box
    xy_threshold = 0.02


    #############
    ### Robot ###
    #############
    # Robot parameters/gains
    joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    ee_body_name = "wrist_3_link"
    robot_init_pos = (0.3, -0.1, 0.0)

    robot_vel_limit = 180.0
    robot_effort_limit = 87.0
    robot_stiffness = 0.0
    robot_damping = 0.0

    robot_initial_joint_pos = [2.5, -2.0, 2.0, -1.5, -1.5, 0.0, 0.0, 0.0] # With gripper joint pos set to 0.0
    robot_reset_joints_pos_range = (1.0, 1.0)
    robot_reset_joints_vel_range = (0.0, 0.0)
    robot_reset_joints_asset_cfg = SceneEntityCfg("robot", joint_names=["wrist_3_joint"]) # "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", 

    # tcp_rand_range_x = (0.004, 0.004) 
    tcp_rand_range_x = (-0.008, 0.008) #(-0.005, 0.005) # (-0.008, 0.008) # (0.0, 0.0) # was +/- 2 cm before
    # tcp_rand_range_y = (-0.004, -0.004)
    tcp_rand_range_y = (-0.008, 0.008) #(-0.005, 0.005) # (-0.008, 0.008) # (0.005, 0.005) # was +/- 2 cm before
    # tcp_rand_range_z = (0.27, 0.27)
    tcp_rand_range_z = (0.0675, 0.07) # (0.068, 0.068) # (0.1, 0.125)    # 7.6 cm is the height for the peg being almost in contact with the hole
    tcp_rand_range_roll = (0.0, 0.0)
    tcp_rand_range_pitch = (math.pi, math.pi)
    tcp_rand_range_yaw = (0.0, 0.0) # (-3.14, 3.14)


    ###############
    ### Gripper ###
    ###############
    # Gripper parameters/gains
    gripper_joint_names = ["joint_left", "joint_right"]
    gripper_body_names = ["finger_left", "finger_right"]
    gripper_vel_limit = 1000000.0
    gripper_effort_limit = 200.0
    gripper_stiffness = 10000000.0
    gripper_damping = 50000.0

    # Randomize gripper finger friction
    gripper_static_friction_distribution_params = (1.4, 1.4)
    gripper_dynamic_friction_distribution_params = (1.4, 1.4)
    gripper_restitution_distribution_params = (0.1, 0.1)
    gripper_randomize_friction_operation = "abs"
    gripper_randomize_friction_distribution = "uniform"
    gripper_randomize_friction_make_consistent = True # Ensure dynamic friction <= static friction

    gripper_offset = [0.0, 0.0, 0.15] # or [0, 0, 0.135]
    gripper_open = [0.0, 0.0]
    gripper_joint_pos_close = [-0.025, -0.025]


    ##############
    ### Object ###
    ##############
    # Object parameters
    object_scale = (0.92, 0.92, 1.0)
    object_init_mass = 0.0025
    object_randomize_mass_range = (0.5, 0.5) # (0.1, 1.0)
    object_randomize_mass_operation = "abs"
    object_randomize_mass_distribution = "uniform"
    object_randomize_mass_recompute_inertia = True
    object_init_pos = (-0.2, 0.0, 0.1)

    # Domain randomize object friction
    object_static_friction_distribution_params = (0.2, 0.2)
    object_dynamic_friction_distribution_params = (1.4, 1.4)
    object_restitution_distribution_params = (0.1, 0.1)
    object_randomize_friction_operation = "abs"
    object_randomize_friction_distribution = "uniform"
    object_randomize_friction_make_consistent = True # Ensure dynamic friction <= static friction

    object_rand_pos_range_x = (-0.0015, 0.0015) # (0.001, 0.001) # (-0.0015, 0.0015) # was +/- 3 mm before
    object_rand_pos_range_z = (0.008, 0.015) #(0.008, 0.012) #(0.01, 0.01) # (0.008, 0.015) # was (0.005, 0.02) before
    object_width = 0.008 # 8 mm
    object_height = 0.05 # 5 cm

    ############
    ### Hole ###
    ############
    # Hole parameters
    hole_init_mass = 10
    hole_init_pos = (-0.2, 0.2, 0.0025)
    hole_height = 0.0275
    hole_randomize_mass_range = (10, 10) # (10, 15)
    hole_randomize_mass_operation = "abs"
    hole_randomize_mass_distribution = "uniform"
    hole_randomize_mass_recompute_inertia = True
    hole_randomize_pose_range_x = (-0.05, 0.05) # (0.0, 0.0) # (-0.05, 0.05) #(0.0, 0.0) #(-0.01, 0.01)
    hole_randomize_pose_range_y = (-0.05, 0.05) # (0.0, 0.0) # (-0.05, 0.05) #(0.0, 0.0) #(-0.01, 0.01)
    hole_randomize_pose_range_z = (0.0, 0.0)
    # hole_randomize_pose_range_yaw = (-math.pi, math.pi)