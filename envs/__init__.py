from gym.envs.registration import register

register(
    id='FixedReacherBulletEnv-v0',
    entry_point='envs.fixed_reacher_env:FixedReacherBulletEnv',
    max_episode_steps=150,
    reward_threshold=18.0,
)
