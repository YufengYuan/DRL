

class BaseAgent:

	def __init__(self, model, lr):
		self.model = model
		self.lr = lr

	def act(self, *args, **kwargs):
		raise NotImplementedError

	def get_q_value(self, *args, **kwargs):
		raise NotImplementedError

	def get_value(self, *args, **kwargs):
		raise NotImplementedError

	def get_pi(self, *args, **kwargs):
		raise NotImplementedError

	def get_logprob(self, *args, **kwargs):
		raise NotImplementedError