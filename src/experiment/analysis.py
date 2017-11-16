import os

from matplotlib import pyplot as plt

from src.experiment.plot import plot_multi_average_reward_over_time, plot_multi_average_actions_over_time, \
	plot_multi_average_food_count_over_time, plot_multi_average_self_collision_death_over_time
from src.util.io import read_model


def plot_model_analysis(models, labels, prefix, params):
	actions_per_episode = []
	rewards_per_episode = []
	food_count_per_episode = []
	self_collision_death_per_episode = []

	x = None

	for model in models:
		actions_per_episode.append(model.actions_per_episode[::10])
		rewards_per_episode.append(model.rewards_per_episode[::10])
		food_count_per_episode.append(model.food_count_per_episode[::10])
		self_collision_death_per_episode.append(model.self_collision_death_per_episode[::10])

		if x is None:
			x = list(range(1, model.actions_per_episode.shape[0] + 1, 10))

	plt.figure()
	plot_multi_average_reward_over_time(x, rewards_per_episode, labels)
	plt.savefig(os.path.join(params.image_output_dir, prefix + "_average_reward_over_time.png"))

	plt.figure()
	plot_multi_average_actions_over_time(x, actions_per_episode, labels)
	plt.savefig(os.path.join(params.image_output_dir, prefix + "_average_actions_over_time.png"))

	plt.figure()
	plot_multi_average_food_count_over_time(x, food_count_per_episode, labels)
	plt.savefig(os.path.join(params.image_output_dir, prefix + "_average_food_count_over_time.png"))

	plt.figure()
	plot_multi_average_self_collision_death_over_time(x, self_collision_death_per_episode, labels)
	plt.savefig(os.path.join(params.image_output_dir, prefix + "_average_self_collision_death_over_time.png"))


def filter_models(models, states, predicate):
	current_models = []
	current_states = []

	for i, state in enumerate(states):
		if predicate(state):
			current_models.append(models[i])
			current_states.append(state)

	return current_models, current_states


def filter_models_with_rewards(models, states, rewards, predicate):
	current_models = []
	current_states = []
	current_rewards = []

	for i, (state, reward) in enumerate(zip(states, rewards)):
		if predicate(state):
			current_models.append(models[i])
			current_states.append(state)
			current_rewards.append(reward)

	return current_models, current_states, current_rewards


def read_models(params):
	filenames = []
	models = []

	for subdir, dirs, files in os.walk(params.model_output_dir):
		for file in files:
			file_path = os.path.join(subdir, file)
			filenames.append(file)
			models.append(read_model(file_path))

	return models, filenames
