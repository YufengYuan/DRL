import numpy as np
from pybullet_envs.gym_manipulator_envs import ReacherBulletEnv
from pybullet_envs.robot_manipulators import Reacher
from pybullet_envs.env_bases import MJCFBaseBulletEnv

class FixedReacher(Reacher):
    def calc_state(self):
        theta, self.theta_dot = self.central_joint.current_position()
        self.theta_dot *= 0.1   # to be consistent w/ how velocity was calculated before this fix
        state = Reacher.calc_state(self)
        state[4:7] = np.array([np.cos(theta), np.sin(theta), self.theta_dot])
        #state[:2] = 0
        #state[2:4] = 0
        #state[:4] += np.random.randn(4) * 0.1
        return state


class FixedReacherBulletEnv(ReacherBulletEnv):
    def __init__(self, render=False):
        self.robot = FixedReacher()
        MJCFBaseBulletEnv.__init__(self, self.robot, render)
