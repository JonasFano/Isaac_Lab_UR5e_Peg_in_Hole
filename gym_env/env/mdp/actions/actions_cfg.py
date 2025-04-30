from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ...controller.impedance_control_cfg import ImpedanceControllerCfg
from .actions import ImpedanceControllerAction

@configclass
class ImpedanceControllerActionCfg(ActionTermCfg):
    """Configuration for impedance controller action term."""

    class_type: type[ActionTerm] = ImpedanceControllerAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions."""

    body_name: str = MISSING
    """Name of the body or frame on which the impedance controller operates."""

    controller_cfg: ImpedanceControllerCfg = MISSING
    """Configuration for the ImpedanceController."""

    action_scale: float = 1.0
    """Scale factor for relative 3D position input. Defaults to 1.0."""

    tcp_offset: list[float] = None
    """Optional fixed offset [x, y, z] from EEF to TCP, expressed in EEF frame."""
