import os
from collections import defaultdict

from matplotlib import pyplot as plt

from src.experiments.plot import plot_multi_average_reward_over_time, plot_multi_average_actions_over_time, \
	plot_multi_average_food_count_over_time, plot_multi_average_self_collision_death_over_time, \
	plot_multi_average_game_score_over_time
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
		food_count_per_episode.append(100 * model.food_count_per_episode[::10])
		self_collision_death_per_episode.append(model.self_collision_death_per_episode[::10])

		if x is None:
			x = list(range(1, model.actions_per_episode.shape[0] + 1, 10))
	#
	# plt.figure()
	# plot_multi_average_reward_over_time(x, rewards_per_episode, labels)
	# plt.savefig(os.path.join(params.image_output_dir, prefix + "_average_reward_over_time.png"))
	#
	# plt.figure()
	# plot_multi_average_actions_over_time(x, actions_per_episode, labels)
	# plt.savefig(os.path.join(params.image_output_dir, prefix + "_average_actions_over_time.png"))

	plt.figure()
	plot_multi_average_game_score_over_time(x, food_count_per_episode, labels)
	plt.savefig(os.path.join(params.image_output_dir, prefix + "_average_game_score_over_time.png"))


# plt.figure()
# plot_multi_average_self_collision_death_over_time(x, self_collision_death_per_episode, labels)
# plt.savefig(os.path.join(params.image_output_dir, prefix + "_average_self_collision_death_over_time.png"))


def read_models(params):
	filenames = []
	models = []

	for subdir, dirs, files in os.walk(params.model_output_dir):
		for file in files:
			file_path = os.path.join(subdir, file)
			filenames.append(file)
			models.append(read_model(file_path))

	return models, filenames


def get_aggregated_models(algorithm, experiment, params, seeds):
	if algorithm not in ("sarsa", "qlearning", "expected_sarsa"):
		raise ValueError("Unknown algorithm.")

	if experiment not in ("reward", "state", "params"):
		raise ValueError("Unknown experiments.")

	filename_models = defaultdict(list)

	for seed in seeds:
		params.seed = seed

		model_output_dir = "../../../models/%s/%s/%i" % (algorithm, experiment, params.seed)
		params.model_output_dir = model_output_dir

		models, filenames = read_models(params)

		for filename, model in zip(filenames, models):
			filename_models[filename].append(model)

	aggregated_models = dict()

	for filename, models in filename_models.items():
		current_model = models[0]

		for model in models[1:]:
			current_model.rewards_per_episode += model.rewards_per_episode
			current_model.actions_per_episode += model.actions_per_episode
			current_model.exploratory_actions_per_episode += model.exploratory_actions_per_episode
			current_model.food_count_per_episode += model.food_count_per_episode
			current_model.self_collision_death_per_episode += model.self_collision_death_per_episode

		current_model.rewards_per_episode = current_model.rewards_per_episode / len(models)
		current_model.actions_per_episode = current_model.actions_per_episode / len(models)
		current_model.exploratory_actions_per_episode = current_model.exploratory_actions_per_episode / len(models)
		current_model.food_count_per_episode = current_model.food_count_per_episode / len(models)
		current_model.self_collision_death_per_episode = current_model.self_collision_death_per_episode / len(models)

		aggregated_models[filename] = current_model

	return aggregated_models
