import numpy as np

from src.core.agent import Agent


class SnakePlayer(Agent):
	def __init__(self):
		self.action = None

	def get_action(self, state):
		return self.action


class SnakeAgent(Agent):
	def __init__(self, policy, action_value_function):
		self.action_values = action_value_function
		self.policy = policy

	def get_action(self, state):
		return self.policy.get_action(state, self.action_values)


class SnakeRandomAgent(Agent):
	def __init__(self, env):
		self.env = env

	def get_action(self, state):
		return np.random.choice(self.env.get_valid_actions(state))
