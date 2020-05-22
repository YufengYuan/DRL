import gym
#from gym.envs.mujoco.reacher import ReacherEnv
import numpy as np
#from mujoco_py.generated import const
import envs
try:
	import pybullet_envs
except ImportError:
	print('Pybullet not found')
import pybullet


class VisualReacher(gym.core.Wrapper):

	def __init__(self, image_size=(480, 480), sampling_interval=2, ignore_target=True):
		"""
		Initialize this wrapper for the original Reacher task
		"""
		super(VisualReacher, self).__init__(env=gym.make('FixedReacherBulletEnv-v0'))
		self.image_size = image_size
		self.ignore_target = ignore_target
		self.image = None
		self.image_offset = 130
		self._render_height = 480
		self._render_width = 480
		self.sampling_interval = sampling_interval
		self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,))
		self.reset()

	def step(self, action):
		"""
		CNN input is set as an attribute for this wrapper for compatibility with Gym interface
		"""
		obs, reward, done, info = super(VisualReacher, self).step(action)
		#self.image = self.render(mode='rgb_array')
		self.image = self.capture_image()
		self.process_image()
		info['target'] = obs[:2]
		info['image'] = self.image
		tx, ty = obs[:2]
		obs[2] += tx
		obs[3] += ty
		return obs[2:], reward, done, info

	def reset(self, **kwargs):
		"""
		Reset the environment and update screenshot
		"""
		#self.env.viewer.cam.lookat = [0, 0, 0]
		obs = super(VisualReacher, self).reset(**kwargs)
		self.image = self.capture_image()
		self.process_image()
		if not hasattr(self, 'visual_space'):
			self.visual_space = gym.spaces.Box(low=-1., high=1., shape=self.image.shape[:2], dtype=np.float32)
		tx, ty = obs[:2]
		obs[2] += tx
		obs[3] += ty
		return obs[2:]

	def setup_camera(self):
		# figure out the position of current camera and fix its position at center
		#self.render(mode='rgb_array')
		#print(dir(self.env))
		#x, y, _ = self.env.viewer.cam.lookat
		#self.env.viewer.move_camera(const.MOUSE_MOVE_H, x, y)
		#self.env.viewer.move_camera(const.MOUSE_ROTATE_V, 0.0, 0.5)
		#x, y, _ = self.env.viewer.cam.lookat
		#assert abs(x) < 3e-3 and abs(y) < 3e-3, f'Failed to setup camera, current posiction: ({x}, {y})!'
		self.env.camera.move_and_look_at(i=0, j=0, k=0, x=0, y=0, z=0)

	def process_image(self):
		size = self.image.shape[0]
		# crop the image based on the image offset
		self.image = self.image[
			self.image_offset: size - self.image_offset,
			self.image_offset: size - self.image_offset,
			:
		    ]
		# Down-sampling the image
		assert self.image.shape[0] % self.sampling_interval == 0, 'Sampling interval not aligned with image size!'
		self.image = self.image[::self.sampling_interval, ::self.sampling_interval, :]
		# Convert uint8 to float32
		#print(self.image)
		#self.image = self.image.astype(np.float32)
		# Normalize pixel values
		#self.image /= 255.
		#print(self.screenshot)


	def capture_image(self):
		#print(dir(self.env.unwrapped))

		camera = self.env.unwrapped._p
		view_matrix = camera.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0],
		                                          distance=1,
		                                          yaw=0,
		                                          pitch=-90,
		                                          roll=0,
		                                          upAxisIndex=2)

		proj_matrix = camera.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(self._render_width) /
                                                     self._render_height,
                                                     nearVal=0.1,
                                                     farVal=100.0)

		(_, _, px, _, _) = camera.getCameraImage(width=self._render_width,
		                                          height=self._render_height,
		                                          viewMatrix=view_matrix,
		                                          projectionMatrix=proj_matrix,
		                                          renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

		rgb_array = np.array(px, dtype=np.uint8)
		rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
		rgb_array = rgb_array[:, :, :3]
		self.image = rgb_array
		return rgb_array

	def get_image(self):
		#return np.rollaxis(self.image, 2, 0)
		return self.image
