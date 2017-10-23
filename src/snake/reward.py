from src.core.reward import Reward


class DefaultSnakeReward(Reward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = env.score
		else:
			self._value = -1

	@property
	def value(self):
		return self._value
