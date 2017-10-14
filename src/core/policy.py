import numpy as np

from src.core.value_function import ActionValueFunction


class Policy(object):
	def __init__(self, env):
		self.env = env

	def get_action(self, state, value_function):
		"""
		Gets the action to perform.

		Parameters
		----------
		state : State
		    The state of the agent.
		value_function : ValueFunction
		    The value function with current estimates.

		Returns
		-------
		action : Action
		    The action to perform.
		"""
		pass


class GreedyPolicy(Policy):
	def get_action(self, state, value_function):
		actions = self.env.get_valid_actions(state)

		if isinstance(value_function, ActionValueFunction):
			return self._get_action_by_action_value(state, actions, value_function)

		return self._get_action_by_state_value(state, actions, value_function)

	@staticmethod
	def _get_action_by_action_value(state, actions, value_function):
		best_action = None
		best_action_value = -np.Inf

		for action in actions:
			action_state_value = value_function.get_value(state, action)

			if action_state_value > best_action_value:
				best_action = best_action
				best_action_value = action_state_value

		return best_action

	def _get_action_by_state_value(self, state, actions, value_function):
		pass


class EpsilonGreedyPolicy(GreedyPolicy):
	def __init__(self, env, epsilon):
		super().__init__(env)

		self.epsilon = epsilon

	def get_action(self, state, value_function):
		actions = self.env.get_valid_actions(state)

		if np.random.uniform(size=1, low=0, high=1) < self.epsilon:
			return actions[np.random.choice(len(actions), size=1)]

		return super().get_action(state, value_function)
