from src.algorithms.td import TemporalDifferenceLearning


class QLearning(TemporalDifferenceLearning):
	def __init__(self, action_value_function, policy, learning_rate, discount_factor):
		self.Q = action_value_function
		self.policy = policy
		self.alpha = learning_rate
		self.gamma = discount_factor

	def train(self, env, n_episodes=100):
		for _ in range(n_episodes):
			state = env.start_new_episode()
			action = self.policy.get_action(state, self.Q)

			while not env.episode_is_done():
				new_state, reward = env.step(action)

				next_action = self.policy.get_action(new_state, self.Q)

				current_value = self.Q.get_value(state, action)

				new_value_est = current_value + self.alpha * (
					reward.value +
					self.gamma * self.Q.get_value(new_state, next_action) -
					current_value
				)

				self.Q.set_value(state, action, new_value_est)

				state = new_state
				action = next_action
