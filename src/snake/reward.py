from src.core.reward import Reward
from src.util.math import manhattan

REWARD_PER_BODY_PART = 100
REWARD_COLLISION = -10000
REWARD_TIME_STEPS = -10
REWARD_DEATH = -10000


class SnakeReward(Reward):
	@property
	def value(self):
		return self._value


class PosTravelPosScore(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = env.score
		else:
			self._value = 1


class NegTravelPosScore(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = env.score
		else:
			self._value = -1


class ZeroTravelPosScore(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = env.score
		else:
			self._value = 0


class PosTravelNegScore(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = -env.score
		else:
			self._value = 1


class NegTravelNegScore(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = -env.score
		else:
			self._value = -1


class ZeroTravelNegScore(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = -env.score
		else:
			self._value = 0


class PosTravelZeroScore(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = 0
		else:
			self._value = 1


class NegTravelZeroScore(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = 0
		else:
			self._value = -1


class ZeroTravelZeroScore(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = 0
		else:
			self._value = 0


class NegDistancePosBodySize(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = REWARD_PER_BODY_PART * len(env.snake.body)
		else:
			self._value = -manhattan(env.snake.head, env.food)


class NegDistanceNegTimeStepPosBodySize(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = REWARD_PER_BODY_PART * len(env.snake.body) - REWARD_TIME_STEPS * pow(env.time_step, 2)
		else:
			self._value = -manhattan(env.snake.head, env.food)


class NegDistanceNegSelfCollisionPosBodySize(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = REWARD_PER_BODY_PART * len(
				env.snake.body) + REWARD_COLLISION * env.death_from_self_collision
		else:
			self._value = -manhattan(env.snake.head, env.food)


class NegDistanceNegBorderCollisionPosBodySize(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = REWARD_PER_BODY_PART * len(env.snake.body) + REWARD_COLLISION * (
				not env.death_from_self_collision)
		else:
			self._value = -manhattan(env.snake.head, env.food)


class NegDistanceNegDeathPosBodySize(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = REWARD_PER_BODY_PART * len(env.snake.body) + REWARD_DEATH
		else:
			self._value = -manhattan(env.snake.head, env.food)


class NegTravelNegBorderCollisionPosScore(SnakeReward):
	def __init__(self, env, state, action, new_state):
		if env.episode_is_done():
			self._value = env.score + REWARD_COLLISION * (not env.death_from_self_collision)
		else:
			self._value = -1
