from math import inf

from src.core.value_function import StateValueFunction


class IterativePolicyEvaluation(object):
	def __init__(self, mdp, discount_factor, threshold):
		self.mdp = mdp
		self.threshold = threshold
		self.discount_factor = discount_factor
		self.value_function = None

	def evaluate(self, policy):
		self.value_function = StateValueFunction(self.mdp.get_num_of_states(), 0)

		previous_delta = inf
		current_delta = 0

		while abs(current_delta - previous_delta) >= self.threshold:
			previous_delta = current_delta
			current_delta = self.update_value_function_and_return_maximum_state_value_difference(policy)

		return self.value_function

	def update_value_function_and_return_maximum_state_value_difference(self, policy):
		delta = -inf

		for state in self.mdp.get_all_states():
			value = self.value_function.get_state_value(state)
			self.value_function.set_value(state, self.compute_expected_state_value(state, policy))
			delta = max(delta, abs(value - self.value_function.get_state_value(state)))

		return delta

	def compute_expected_state_value(self, state, policy):
		expected_value = 0

		for action in self.mdp.get_actions(state):
			action_prob = policy.get_action_prob(action, state)
			reward = self.compute_expected_state_action_reward(state, action)
			expected_value += action_prob * reward

		return expected_value

	def compute_expected_state_action_reward(self, state, action):
		expected_reward = 0

		for new_state, reward in self.mdp.get_state_reward_pairs(state, action):
			expected_reward += reward + \
							   self.discount_factor.get_discount_factor(new_state) * \
							   self.value_function.get_state_value(new_state) * \
							   self.mdp.get_state_transition_prob(state, action, new_state)

		return expected_reward
