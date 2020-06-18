import numpy as np
from pybullet_envs.gym_manipulator_envs import ReacherBulletEnv
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv


REACHER_THRESHOLD = -1.0 # -100 * np.linalg.norm(self.to_target_vec)


class SparseReacherBulletEnv(ReacherBulletEnv):

	def __init__(self, render=False):
		super(SparseReacherBulletEnv, self).__init__(render)

	def step(self, a):
		assert (not self.scene.multiplayer)
		self.robot.apply_action(a)
		self.scene.global_step()

		state = self.robot.calc_state()  # sets self.to_target_vec

		potential_old = self.potential
		self.potential = self.robot.calc_potential()

		electricity_cost = (
				-0.10 * (np.abs(a[0] * self.robot.theta_dot) + np.abs(a[1] * self.robot.gamma_dot)
				         )  # work torque*angular_velocity
				- 0.01 * (np.abs(a[0]) + np.abs(a[1]))  # stall torque require some energy
		)
		stuck_joint_cost = -0.1 if np.abs(np.abs(self.robot.gamma) - 1) < 0.01 else 0.0
		self.rewards = [
			float(self.potential - potential_old),
			float(electricity_cost),
			float(stuck_joint_cost)
		]
		self.HUD(state, a, False)
		#return state, float(self.potential > REACHER_THRESHOLD), False, {}
		# Replace the original dense reward with sparse reward
		return state, float(self.potential > REACHER_THRESHOLD), False, {}
