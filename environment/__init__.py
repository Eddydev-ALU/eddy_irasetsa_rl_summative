from gymnasium.envs.registration import register

from environment.custom_env import ACTION_NAMES, GUIDANCE_TEXT, OBS_NAMES, PLANES, EnvConfig, FetalScanNavEnv

register(id="FetalScanNav-v0", entry_point="environment.custom_env:FetalScanNavEnv")

__all__ = ["FetalScanNavEnv", "EnvConfig", "ACTION_NAMES", "GUIDANCE_TEXT", "OBS_NAMES", "PLANES"]
