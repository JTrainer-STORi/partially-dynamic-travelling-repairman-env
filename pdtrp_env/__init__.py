from gymnasium.envs.registration import register

register(
    id="pdtrp_env/patially-dynamic-travelling-repairman-v0",
    entry_point="pdtrp_env.envs.partially_dynamic_travelling_repairman:PartiallyDynamicTravellingRepairmanEnv",
)
