from gym.envs.registration import register
from envs.visual_reacher_bullet import VisualReacher

register(
    id='FixedReacherBulletEnv-v0',
    entry_point='envs.fixed_reacher_env:FixedReacherBulletEnv',
    max_episode_steps=150,
    reward_threshold=18.0,
)

register(
    id='VisualReacherBulletEnv-v0',
    entry_point='envs.visual_reacher_bullet:VisualReacher',
    max_episode_steps=150,
    reward_threshold=18.0
)

register(
    id='ModifiedMountainCar-v0',
    #entry_point='envs.visual_reacher_bullet:VisualReacher',
    entry_point='envs.modified_mountain_car:ModifiedMoutainCar',
    max_episode_steps=200,
    reward_threshold=200
)

