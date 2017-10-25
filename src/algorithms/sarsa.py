from src.algorithms.base import EpisodicTDLearner


class Sarsa(EpisodicTDLearner):
	def compute_estimate(self, state, action, new_state, reward, next_action):
		current_value = self.Q.get_value(state, action)

		new_estimate = current_value + self.alpha(state) * (
			reward.value +
			self.gamma(state) * self.Q.get_value(new_state, next_action) -
			current_value
		)

		return new_estimate


class ExpectedSarsa(EpisodicTDLearner):
	def compute_estimate(self, state, action, new_state, reward, next_action):
		pass
