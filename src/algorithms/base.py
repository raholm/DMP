import numpy as np


class EpisodicTDLearner(object):
	def __init__(self, action_value_function, policy, learning_rate, discount_factor):
		self.Q = action_value_function
		self.policy = policy
		self.alpha = learning_rate
		self.gamma = discount_factor

		self.rewards_per_episode = None
		self.actions_per_episode = None
		self.exploratory_actions_per_episode = None
		self.food_count_per_episode = None
		self.self_collision_death_per_episode = None

	def train(self, env, n_episodes):
		self.rewards_per_episode = np.zeros(n_episodes, dtype=np.int)
		self.actions_per_episode = np.zeros(n_episodes, dtype=np.uint)
		self.exploratory_actions_per_episode = np.zeros(n_episodes, dtype=np.uint)
		self.food_count_per_episode = np.zeros(n_episodes, dtype=np.uint8)
		self.self_collision_death_per_episode = np.zeros(n_episodes, dtype=np.bool)

		for iteration in range(n_episodes):
			if iteration % 10000 == 0:
				print("Iteration:", iteration)

			state = env.start_new_episode()
			action = self.policy.get_action(state, self.Q)

			while not env.episode_is_done():
				new_state, reward = env.step(action)
				next_action = self.policy.get_action(new_state, self.Q)

				new_estimate = self.compute_estimate(state, action, new_state,
													 reward, next_action)

				self.Q.set_value(state, action, new_estimate)

				state = new_state
				action = next_action

				self.rewards_per_episode[iteration] += reward.value
				self.actions_per_episode[iteration] += 1
				self.exploratory_actions_per_episode[iteration] += self.policy.exploratory

			self.food_count_per_episode[iteration] = env.food_count
			self.self_collision_death_per_episode[iteration] = env.death_from_self_collision

		return self

	def compute_estimate(self, state, action, new_state, reward, next_action):
		current_value = self.Q.get_value(state, action)

		new_estimate = current_value + self.alpha(state) * (
			reward.value +
			self.gamma(state) * self.compute_next_value_estimate(state, action, new_state, next_action) -
			current_value
		)

		return new_estimate

	def compute_next_value_estimate(self, state, action, new_state, next_action):
		raise NotImplementedError
