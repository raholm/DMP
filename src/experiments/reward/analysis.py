import numpy as np

from src.experiments.analysis import plot_model_analysis, read_models, get_aggregated_models
from src.experiments.params import ExperimentParameters
from src.experiments.reward.params import get_reward_seeds
from src.util.math import compute_mean_over_time
from src.util.util import filter_models_by_state, filter_models_with_rewards_by_state


def analyze_board_state_models(models, states, rewards, params):
	current_models, current_states, current_rewards = \
		filter_models_with_rewards_by_state(models, states, rewards,
											lambda state: state.startswith("Board"))
	plot_model_analysis(current_models, current_rewards, "board_state", params)


def analyze_snake_food_state_models(models, states, rewards, params):
	current_models, current_states, current_rewards = \
		filter_models_with_rewards_by_state(models, states, rewards,
											lambda state: state.startswith("SnakeFood"))
	plot_model_analysis(current_models, current_rewards, "snake_food_state", params)


def analyze_directional_state_models(models, states, rewards, params):
	current_models, current_states, current_rewards = \
		filter_models_with_rewards_by_state(models, states, rewards,
											lambda state: state.startswith("Directional") and
														  not state.startswith("DirectionalDistance"))
	plot_model_analysis(current_models, current_rewards, "directional_state", params)


def analyze_directional_distance_state_models(models, states, rewards, params):
	current_models, current_states, current_rewards = \
		filter_models_with_rewards_by_state(models, states, rewards,
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

def analyze_models_test(params):
	models, filenames = read_models(params)

	states = []
	rewards = []

	for filename in filenames:
		states.append(filename.split("_")[0])
		rewards.append(filename.split("_")[1])

	data = []

	for model, state, reward in zip(models, states, rewards):
		rewards = model.run(params.env, 1000)
		mean_reward = np.mean(rewards)
		sd_reward = np.std(rewards)

		data.append((state, reward, mean_reward, sd_reward))


def analyze_aggregated_models(filenames, models, params):
	order = list(np.argsort(filenames))
	filenames = [filenames[i] for i in order]
	models = [models[i] for i in order]

	states = []
	rewards = []

	for filename in filenames:
		states.append(filename.split("_")[0])
		rewards.append(filename.split("_")[1])

	analyze_board_state_models(models, states, rewards, params)
	# analyze_directional_state_models(models, states, rewards, params)


def analyze_aggregated_reward_food_count_correlations(algorithm):
	def get_corr_coefs(models):
		corr_coefs = []

		for model in models:
			average_reward_over_time = compute_mean_over_time(model.rewards_per_episode)
			average_food_count_over_time = compute_mean_over_time(model.food_count_per_episode)

			if np.all(average_reward_over_time == 0):
				continue

			corr_coef = np.corrcoef(average_reward_over_time,
									average_food_count_over_time)
			corr_coefs.append(corr_coef)

		return corr_coefs

	exp_params = ExperimentParameters()
	exp_params.seed = get_reward_seeds()[0]

	aggregated_models = get_aggregated_models(algorithm, "reward", exp_params, get_reward_seeds())

	states = [filename.split("_")[0] for filename in aggregated_models.keys()]
	models = list(aggregated_models.values())

	current_models, _ = \
		filter_models_by_state(models, states,
							   lambda state: state.startswith("Board"))

	corr_coefs = get_corr_coefs(current_models)

	print("Board State")
	print("Average correlation:", np.mean(corr_coefs))
	print("Average absolute correlation:", np.mean(np.abs(corr_coefs)))

	current_models, _ = \
		filter_models_by_state(models, states,
							   lambda state: state.startswith("DirectionalDistance"))

	corr_coefs = get_corr_coefs(current_models)

	print("Directional Distance State")
	print("Average correlation:", np.mean(corr_coefs))
	print("Average absolute correlation:", np.mean(np.abs(corr_coefs)))
