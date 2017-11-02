import numpy as np

from src.core.value_function import ActionValueFunction


class Policy(object):
	def __init__(self, env):
		self.env = env
		self.exploratory = False

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
		raise NotImplementedError

	def get_optimal_action(self, state, value_function):
		"""
		Gets the optimal action in terms of the value function.

		Parameters
		----------
		state : State
		    The state of the agent.
		value_function : ValueFunction
		    The value function with current estimates.

		Returns
		-------
		action : Action
		    The optimal greedy action.
		"""
		raise NotImplementedError

	def get_action_probabilities(self, state, value_function):
		"""
		Gets the action probabilities in the state.

		Parameters
		----------
		state : State
		    The state of the agent.
		value_function : ValueFunction
		    The value function with current estimates.

		Returns
		-------
		actions : List of Action
		    A list of available actions in the state.

		probabilities : Array of Float
		    A list of probabilities of taking the corresponding action in the state.
		"""
		raise NotImplementedError


class GreedyPolicy(Policy):
	def get_action(self, state, value_function):
		return self.get_optimal_action(state, value_function)

	def get_optimal_action(self, state, value_function):
		actions = self.env.get_valid_actions(state)
		return self._get_optimal_action(state, actions, value_function)

	def get_action_probabilities(self, state, value_function):
		actions = self.env.get_valid_actions(state)
		probabilities = np.zeros(len(actions))

		optimal_action = self._get_optimal_action(state, actions, value_function)

		for i, action in enumerate(actions):
			if action == optimal_action:
				probabilities[i] = 1

		return actions, probabilities

	def _get_optimal_action(self, state, actions, value_function):
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
				best_action = action
				best_action_value = action_state_value

		return best_action

	def _get_action_by_state_value(self, state, actions, value_function):
		raise NotImplementedError


class EpsilonGreedyPolicy(GreedyPolicy):
	def __init__(self, env, epsilon):
		super().__init__(env)

		self.epsilon = epsilon

	def get_action(self, state, value_function):
		self.exploratory = False

		if np.random.uniform(size=1, low=0, high=1) < self.epsilon:
			self.exploratory = True
			actions = self.env.get_valid_actions(state)
			return np.random.choice(actions, size=1)[0]

		return super().get_action(state, value_function)

	def get_optimal_action(self, state, value_function):
		return super().get_optimal_action(state, value_function)

	def get_action_probabilities(self, state, value_function):
		actions = self.env.get_valid_actions(state)
		n = len(actions)
		probabilities = np.full(n, self.epsilon * (1 / n))

		optimal_action = self._get_optimal_action(state, actions, value_function)

		for i, action in enumerate(actions):
			if action == optimal_action:
				probabilities[i] += 1 - self.epsilon

		return actions, probabilities
