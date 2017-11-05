from src.core.reward import Reward


class SnakeReward(Reward):
	@property
	def value(self):
		return self._value


class PositiveTravelPositiveFood(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = env.score
		else:
			self._value = 1


class NegativeTravelPositiveFood(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = env.score
		else:
			self._value = -1


class ZeroTravelPositiveFood(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = env.score
		else:
			self._value = 0


class PositiveTravelNegativeFood(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = -env.score
		else:
			self._value = 1


class NegativeTravelNegativeFood(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = -env.score
		else:
			self._value = -1


class ZeroTravelNegativeFood(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = -env.score
		else:
			self._value = 0


class PositiveTravelZeroFood(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = 0
		else:
			self._value = 1


class NegativeTravelZeroFood(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = 0
		else:
			self._value = -1


class ZeroTravelZeroFood(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = 0
		else:
			self._value = 0
