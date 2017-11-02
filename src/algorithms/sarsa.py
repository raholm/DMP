from src.algorithms.base import EpisodicTDLearner


class Sarsa(EpisodicTDLearner):
	def compute_next_value_estimate(self, state, action, new_state, next_action):
		return self.Q.get_value(new_state, next_action)


class ExpectedSarsa(EpisodicTDLearner):
	def compute_next_value_estimate(self, state, action, new_state, next_action):
		actions, probabilities = self.policy.get_action_probabilities(new_state)
		estimate = 0

		for action, probability in zip(actions, probabilities):
			estimate += probability * self.Q.get_value(new_state, action)

		return estimate
