from dataclasses import dataclass
from isaaclab.utils import configclass
from .impedance_control import ImpedanceController


@configclass
class ImpedanceControllerCfg:
    """Configuration for the Simple Impedance Controller.

    Supports 3D position impedance control with optional gravity compensation and inertial decoupling.
    """

    class_type: type = ImpedanceController
    gravity_compensation: bool = False
    coriolis_centrifugal_compensation: bool = False
    inertial_dynamics_decoupling: bool = False
    max_torque_clamping: list[float] = None # For each joint one max torque 
    stiffness: list[float] = None  # 6D stiffness: [x, y, z, rx, ry, rz]
    damping: list[float] | None = None  # Optional 6D damping
