import math
from isaaclab.managers import SceneEntityCfg

class TaskParams:
    #################################
    ### General Simulation Params ###
    #################################
    # decimation = 2
    decimation = 20 # 50 Hz control frequency
    episode_length_s = 100.0 # 10.0  # 10.0 # 0.5 # 5.0
    # dt = 1/100
    dt = 1/1000 # 500 Hz simulation frequency
    render_interval = 10
    gravity = [0.0, 0.0, -9.81]


    #########################
    ### Impedance Control ###
    #########################
    command_type = "pose"
    use_relative_mode = True
    ik_method = "dls"
    action_scale= 0.8 # 0.005 # 0.0

    gravity_compensation = True
    coriolis_centrifugal_compensation = True
    inertial_dynamics_decoupling = True
    max_torque_clamping = None # Array of max torques to clamp computed torques - no clamping if None
    stiffness = [400, 400, 500, 300, 300, 300] # Array of Kd stiffness for x y z rx ry rz
    damping = None # None = Critically damped
    damping_ratio = 1
    damping_ratio_z = 1


    ################
    # Observations #
    ################
    tcp_pose_unoise_min = -0.0001 # 0.1 mm
    tcp_pose_unoise_max = 0.0001 # 0.1 mm

    noise_std_hole_pose = 0.0 # 0.0025 # 2.5 mm


    #########
    # Event #
    #########
    ik_max_iters = 30
    pos_error_threshold = 1e-3
    angle_error_threshold = 1e-3
    levenberg_marquardt_lambda = 0.01


    ##############
    ### Reward ###
    ##############
    action_rate_weight = -1e-4
    action_rate_curriculum_weight = -1e-1
    action_rate_curriculum_num_steps = 50000

    # Keypoint distance/
    num_keypoints = 4
    coarse_kernel_a = 50
    coarse_kernel_b = 2
    keypoint_distance_coarse_weight = 5.0

    fine_kernel_a = 100
    fine_kernel_b = 0
    keypoint_distance_fine_weight = 5.0

    # Is peg centered
    is_peg_centered_xy_threshold = 0.0025 # 2.5 mm
    is_peg_centered_z_threshold = 0.09 # 8 cm
    is_peg_centered_z_variability = 0.002 # 2 mm
    is_peg_centered_weight = 20.0

    # Is peg inserted
    is_peg_inserted_weight = 100.0


    ###################
    ### Termination ###
    ###################
    object_dropping_min_height = -0.05 


    #############
    ### Robot ###
    #############
    # Robot parameters/gains
    joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
    ee_body_name = "panda_hand"

    panda_arm_stiffness = 0.0
    panda_arm_damping = 0.0
    panda_arm_friction = 0.75
    panda_arm_amature = 0.0
    panda_arm1_effort_limit = 87
    panda_arm2_effort_limit = 12
    panda_arm1_velocity_limit = 124.6
    panda_arm2_velocity_limit = 149.5

    robot_initial_joint_pos = [1.8, -0.2, 0.0, -2.4, 0.35, 2.3, 0.8, 0.04] # With gripper joint pos set to 0.0
    robot_init_pos = (0.3, -0.1, 0.0)
    robot_init_rot = (1.0, 0.0, 0.0, 0.0)

    tcp_rand_range_x = (-0.0, 0.0) # was +/- 2 cm before
    tcp_rand_range_y = (-0.001, -0.001) #(-0.005, 0.005) # was +/- 2 cm before
    # tcp_rand_range_z = (0.27, 0.27) # (0.1, 0.125)    # 7.6 cm is the height for the peg being almost in contact with the hole
    tcp_rand_range_z = (0.07, 0.07) # (0.1, 0.125)    # 7.6 cm is the height for the peg being almost in contact with the hole
    tcp_rand_range_roll = (0.0, 0.0)
    tcp_rand_range_pitch = (math.pi, math.pi)
    tcp_rand_range_yaw = (0.0, 0.0) # (-3.14, 3.14)


    ###############
    ### Gripper ###
    ###############
    # Gripper parameters/gains
    gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
    gripper_body_names = ["panda_leftfinger", "panda_rightfinger"]

    gripper_effort_limit = 40.0
    gripper_velocity_limit = 0.04
    gripper_stiffness = 10000000.0
    gripper_damping = 5000.0
    gripper_friction = 0.9
    gripper_armature = 0.0

    # Randomize gripper finger friction
    gripper_static_friction_distribution_params = (1.4, 1.4)
    gripper_dynamic_friction_distribution_params = (1.4, 1.4)
    gripper_restitution_distribution_params = (0.1, 0.1)
    gripper_randomize_friction_operation = "abs"
    gripper_randomize_friction_distribution = "uniform"
    gripper_randomize_friction_make_consistent = True # Ensure dynamic friction <= static friction

    gripper_offset = [0.0, 0.0, 0.107]
    gripper_open = [0.04, 0.04]
    gripper_joint_pos_close = [0.0, 0.0]


    ##############
    ### Object ###
    ##############
    # Object parameters
    object_scale = (0.92, 0.92, 1.0)
    object_init_mass = 0.2
    object_init_pos = (-0.2, 0.0, 0.1)

    # Domain randomize object friction
    object_static_friction_distribution_params = (0.4, 0.4)
    object_dynamic_friction_distribution_params = (1.4, 1.4)
    object_restitution_distribution_params = (0.2, 0.2)
    object_randomize_friction_operation = "abs"
    object_randomize_friction_distribution = "uniform"
    object_randomize_friction_make_consistent = True # Ensure dynamic friction <= static friction

    object_rand_pos_range_x = (-0.00, 0.00) # was +/- 3 mm before
    object_rand_pos_range_z = (0.01, 0.01) # was (0.005, 0.02) before
    object_width = 0.008 # 8 mm
    object_height = 0.05 # 5 cm

    ############
    ### Hole ###
    ############
    # Hole parameters
    hole_init_mass = 10
    hole_init_pos = (-0.2, 0.2, 0.0025)
    hole_randomize_pose_range_x = (0.0, 0.0) #(-0.05, 0.05) #(0.0, 0.0) #(-0.01, 0.01)
    hole_randomize_pose_range_y = (0.0, 0.0) #(-0.05, 0.05) #(0.0, 0.0) #(-0.01, 0.01)
    hole_randomize_pose_range_z = (0.0, 0.0)
    # hole_randomize_pose_range_yaw = (-math.pi, math.pi)