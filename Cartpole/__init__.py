from gym.envs.registration import register

register(
    id='CartPoleLQREnv-v0',
    entry_point='env.cartpole_lqr_env:CartPoleLQREnv'
)

register(
    id='NoisyCartPoleLQREnv-v0',
    entry_point='env.cartpole_lqr_env:CartPoleLQREnv',
    kwargs = {"noisy": True}
)

register(
    id='CartPoleILQREnv-v0',
    entry_point='env.cartpole_ilqr_env:CartPoleILQREnv',
    timestep_limit=200,
    reward_threshold=195.0,
)