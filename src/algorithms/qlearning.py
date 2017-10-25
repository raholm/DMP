from src.algorithms.base import EpisodicTDLearner


class QLearning(EpisodicTDLearner):
	def compute_estimate(self, state, action, new_state, reward, next_action):
		optimal_action = self.policy.get_optimal_action(new_state, self.Q)
		current_value = self.Q.get_value(state, action)

		new_estimate = current_value + self.alpha(state) * (
			reward.value +
			self.gamma(state) * self.Q.get_value(new_state, optimal_action) -
			current_value
		)

		return new_estimate
