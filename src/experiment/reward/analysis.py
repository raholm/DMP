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
	analyze_snake_food_state_models(models, states, rewards, params)
	analyze_directional_state_models(models, states, rewards, params)
	analyze_directional_distance_state_models(models, states, rewards, params)
