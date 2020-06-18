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
    id='SparseReacherBulletEnv-v0',
    entry_point='envs.sparse_envs:SparseReacherBulletEnv',
    max_episode_steps=150,
    reward_threshold=18.0,
)

