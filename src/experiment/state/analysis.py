from src.experiment.analysis import filter_models, plot_model_analysis, read_models


def analyze_board_state_models(models, states, params):
	current_models, current_states = \
		filter_models(models, states,
					  lambda state: state.startswith("Board"))
	plot_model_analysis(current_models, current_states, "board_state", params)


def analyze_snake_food_state_models(models, states, params):
	current_models, current_states = \
		filter_models(models, states,
					  lambda state: state.startswith("SnakeFood"))
	plot_model_analysis(current_models, current_states, "snake_food_state", params)


def analyze_directional_state_models(models, states, params):
	current_models, current_states = \
		filter_models(models, states,
					  lambda state: state.startswith("Directional") and
									not state.startswith("DirectionalDistance"))
	plot_model_analysis(current_models, current_states, "directional_state", params)


def analyze_directional_distance_state_models(models, states, params):
	current_models, current_states = \
		filter_models(models, states,
					  lambda state: state.startswith("DirectionalDistance"))
	plot_model_analysis(current_models, current_states, "directional_distance_state", params)


def analyze_models(params):
	models, filenames = read_models(params)

	states = []

	for filename in filenames:
		states.append(filename.split("_")[0])

	analyze_board_state_models(models, states, params)
	analyze_snake_food_state_models(models, states, params)
	analyze_directional_state_models(models, states, params)
	analyze_directional_distance_state_models(models, states, params)
