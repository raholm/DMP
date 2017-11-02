from src.algorithms.base import EpisodicTDLearner


class QLearning(EpisodicTDLearner):
	def compute_next_value_estimate(self, state, action, new_state, next_action):
		optimal_action = self.policy.get_optimal_action(new_state, self.Q)
		return self.Q.get_value(new_state, optimal_action)
