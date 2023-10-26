from gymnasium.envs.registration import register
from .MaintenanceConfig import get_quay_wall_config
from .Maintenance_Gym import MaintenanceEnv

register(
    id='Maintenance-quay-wall-v0',
    entry_point='modcmac_code.environments.Maintenance_Gym:MaintenanceEnv',
    kwargs=get_quay_wall_config()
)
