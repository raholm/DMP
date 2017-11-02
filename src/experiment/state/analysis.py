import os

from matplotlib import pyplot as plt

from src.experiment.plot import plot_multi_average_reward_over_time, plot_multi_average_actions_over_time
from src.util.io import read_learner


def read_models(params):
	filenames = []
	models = []

	for subdir, dirs, files in os.walk(params.model_output_dir):
		for file in files:
			file_path = os.path.join(subdir, file)
			filenames.append(file)
			models.append(read_learner(file_path))

	return models, filenames


def analyze_state_models(models, states):
	actions_per_episode = []
	rewards_per_episode = []

	for model in models:
		actions_per_episode.append(model.actions_per_episode)
		rewards_per_episode.append(model.rewards_per_episode)

	plt.figure(1)

	plt.subplot(121)
	plot_multi_average_reward_over_time(rewards_per_episode, states)

	plt.subplot(122)
	plot_multi_average_actions_over_time(actions_per_episode, states)

	plt.show()


def filter_state_models(models, states, predicate):
	current_models = []
	current_states = []

	for i, state in enumerate(states):
		if predicate(state):
			current_models.append(models[i])
			current_states.append(state)

	return current_models, current_states


def analyze_board_state_models(models, states):
	current_models, current_states = \
		filter_state_models(models, states,
							lambda state: state.startswith("Board"))
	analyze_state_models(current_models, current_states)


def analyze_snake_food_state_models(models, states):
	current_models, current_states = \
		filter_state_models(models, states,
							lambda state: state.startswith("SnakeFood"))
	analyze_state_models(current_models, current_states)


def analyze_directional_state_models(models, states):
	current_models, current_states = \
		filter_state_models(models, states,
							lambda state: state.startswith("Directional") and
										  not state.startswith("DirectionalDistance"))
	analyze_state_models(current_models, current_states)


def analyze_directional_distance_state_models(models, states):
	current_models, current_states = \
		filter_state_models(models, states,
							lambda state: state.startswith("DirectionalDistance"))
	analyze_state_models(current_models, current_states)


def analyze_models(params):
	models, filenames = read_models(params)

	states = []

	for filename in filenames:
		states.append(filename.split("_")[0])

	analyze_board_state_models(models, states)
	analyze_snake_food_state_models(models, states)
	analyze_directional_state_models(models, states)
	analyze_directional_distance_state_models(models, states)