import os
from timeit import default_timer as timer

import numpy as np

from src.snake.state import BoardState, BoardDimensionState, BoardScoreState, BoardDimensionScoreState, SnakeFoodState, \
	SnakeFoodDimensionState, SnakeFoodScoreState, SnakeFoodDimensionScoreState, DirectionalState, \
	DirectionalDimensionState, DirectionalScoreState, DirectionalDimensionScoreState, DirectionalDistanceState, \
	DirectionalDistanceDimensionState, DirectionalDistanceScoreState, DirectionalDistanceDimensionScoreState
from src.util.io import write_learner


def train_and_store_model(params):
	learner = params.model_class(action_value_function=params.model_params.value_function,
								 policy=params.model_params.policy,
								 learning_rate=params.model_params.learning_rate,
								 discount_factor=params.model_params.discount_factor)

	start = timer()

	learner.train(params.env, params.model_params.train_episodes)

	print("Training time:", timer() - start)

	write_learner(learner, os.path.join(params.model_output_dir, "%s.p" % params.model_params.file_str))

	return learner


def train_state_models(states, params):
	for state in states:
		params.model_params.state = state
		train_and_store_model(params)


def train_board_state_models(params):
	states = [BoardState, BoardDimensionState,
			  BoardScoreState, BoardDimensionScoreState]
	train_state_models(states, params)


def train_snake_food_state_models(params):
	states = [SnakeFoodState, SnakeFoodDimensionState,
			  SnakeFoodScoreState, SnakeFoodDimensionScoreState]
	train_state_models(states, params)


def train_directional_state_models(params):
	states = [DirectionalState, DirectionalDimensionState,
			  DirectionalScoreState, DirectionalDimensionScoreState]
	train_state_models(states, params)


def train_directional_distance_state_models(params):
	states = [DirectionalDistanceState, DirectionalDistanceDimensionState,
			  DirectionalDistanceScoreState, DirectionalDistanceDimensionScoreState]
	train_state_models(states, params)


def train_models(params):
	np.random.seed(params.seed)

	start = timer()

	train_board_state_models(params)
	train_snake_food_state_models(params)
	train_directional_state_models(params)
	train_directional_distance_state_models(params)

	print("Elapsed time:", timer() - start)
