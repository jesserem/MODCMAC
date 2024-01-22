from gymnasium.envs.registration import register
from .MaintenanceConfig import get_quay_wall_config, simple_setup_new_det, get_quay_wall_config_complex
from .Maintenance_Gym import MaintenanceEnv

register(
    id='Maintenance-quay-wall-v0',
    entry_point='modcmac_code.environments.Maintenance_Gym:MaintenanceEnv',
    kwargs=get_quay_wall_config()
)

register(
    id='Maintenance-simple-new-det-v0',
    entry_point='modcmac_code.environments.Maintenance_Gym:MaintenanceEnv',
    kwargs=simple_setup_new_det()
)

register(
    id='Maintenance-quay-wall-complex-v0',
    entry_point='modcmac_code.environments.Maintenance_Gym:MaintenanceEnv',
    kwargs=get_quay_wall_config_complex()
)
