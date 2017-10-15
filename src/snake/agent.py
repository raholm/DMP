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
