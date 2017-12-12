from collections import defaultdict

from src.experiment.analysis import plot_model_analysis, read_models, filter_models_with_rewards


def analyze_board_state_models(models, states, rewards, params):
	current_models, current_states, current_rewards = \
		filter_models_with_rewards(models, states, rewards,
								   lambda state: state.startswith("Board"))
	plot_model_analysis(current_models, current_rewards, "board_state", params)


def analyze_snake_food_state_models(models, states, rewards, params):
	current_models, current_states, current_rewards = \
		filter_models_with_rewards(models, states, rewards,
								   lambda state: state.startswith("SnakeFood"))
	plot_model_analysis(current_models, current_rewards, "snake_food_state", params)


def analyze_directional_state_models(models, states, rewards, params):
	current_models, current_states, current_rewards = \
		filter_models_with_rewards(models, states, rewards,
								   lambda state: state.startswith("Directional") and
												 not state.startswith("DirectionalDistance"))
	plot_model_analysis(current_models, current_rewards, "directional_state", params)


def analyze_directional_distance_state_models(models, states, rewards, params):
	current_models, current_states, current_rewards = \
		filter_models_with_rewards(models, states, rewards,
								   lambda state: state.startswith("DirectionalDistance"))
	plot_model_analysis(current_models, current_rewards, "directional_distance_state", params)


def analyze_models(params):
	models, filenames = read_models(params)

	states = []
	rewards = []

	for filename in filenames:
		states.append(filename.split("_")[0])
		rewards.append(filename.split("_")[1])

	analyze_board_state_models(models, states, rewards, params)
	# analyze_snake_food_state_models(models, states, rewards, params)
	analyze_directional_state_models(models, states, rewards, params)
	# analyze_directional_distance_state_models(models, states, rewards, params)


def get_aggregated_models(algorithm, experiment, params, seeds):
	if algorithm not in ("sarsa", "qlearning", "expected_sarsa"):
		raise ValueError("Unknown algorithm.")

	if experiment not in ("reward", "state", "params"):
		raise ValueError("Unknown experiment.")

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


def analyze_aggregated_models(filenames, models, params):
	states = []
	rewards = []

	for filename in filenames:
		states.append(filename.split("_")[0])
		rewards.append(filename.split("_")[1])

	analyze_board_state_models(models, states, rewards, params)
	analyze_directional_distance_state_models(models, states, rewards, params)
