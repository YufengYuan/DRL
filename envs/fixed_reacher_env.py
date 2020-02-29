import numpy as np

from pybullet_envs.gym_manipulator_envs import ReacherBulletEnv
from pybullet_envs.robot_manipulators import Reacher
from pybullet_envs.env_bases import MJCFBaseBulletEnv

class FixedReacher(Reacher):
    def calc_state(self):
        #theta, self.theta_dot = self.central_joint.current_position()
        #self.theta_dot *= 0.1   # to be consistent w/ how velocity was calculated before this fix
        #state = Reacher.calc_state(self)
        #state[4:7] = np.array([np.cos(theta), np.sin(theta), self.theta_dot])
        #return state
        theta, self.theta_dot = self.central_joint.current_position()
        self.gamma, self.gamma_dot = self.elbow_joint.current_position()
        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()
        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())
        return np.array([
            target_x,
            target_y,
            self.to_target_vec[0],
            self.to_target_vec[1],
            np.cos(theta*0.1),
            np.sin(theta*0.1),
            self.theta_dot*0.1,
            self.gamma*0.1,
            self.gamma_dot*0.1,
        ])

class FixedReacherBulletEnv(ReacherBulletEnv):
    def __init__(self, render=False):
        self.robot = FixedReacher()
        MJCFBaseBulletEnv.__init__(self, self.robot, render)
