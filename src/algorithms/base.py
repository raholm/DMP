import numpy as np

from src.core.experience_buffer import ExperienceBuffer


class EpisodicTDLearner(object):
	def __init__(self, action_value_function, policy,
				 learning_rate, discount_factor,
				 experience_replay=False,
				 experience_replay_update_freq=2,
				 experience_replay_batch_size=10):
		self.Q = action_value_function
		self.policy = policy
		self.alpha = learning_rate
		self.gamma = discount_factor

		self.experience_replay = experience_replay
		self.experience_replay_update_freq = experience_replay_update_freq
		self.experience_replay_batch_size = experience_replay_batch_size

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

		if self.experience_replay:
			experience_buffer = ExperienceBuffer()

		for iteration in range(n_episodes):
			if (iteration + 1) % 10000 == 0:
				print("Iteration:", iteration + 1)

			state = env.start_new_episode()
			action = self.policy.get_action(state, self.Q)

			while not env.episode_is_done():
				new_state, reward = env.step(action)
				next_action = self.policy.get_action(new_state, self.Q)

				new_estimate = self.compute_estimate(state, action, new_state,
													 reward, next_action)

				self.Q.set_value(state, action, new_estimate)

				if self.experience_replay:
					experience_buffer.add([state, action, reward,
										   new_state, env.episode_is_done()])

					if (iteration + 1) % self.experience_replay_update_freq == 0:
						self._train_on_experience_batch(experience_buffer)

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

	def _train_on_experience_batch(self, experience_buffer):
		train_batch = experience_buffer.sample(self.experience_replay_batch_size)

		for (state, action, reward, new_state, done) in train_batch:
			next_action = self.policy.get_action(new_state, self.Q)
			new_estimate = self.compute_estimate(state, action, new_state,
												 reward, next_action)
			self.Q.set_value(state, action, new_estimate)
