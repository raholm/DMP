from timeit import default_timer as timer

import numpy as np

from src.experiment.train import train_and_store_model
from src.snake.state import BoardState, BoardDimensionState, BoardScoreState, BoardDimensionScoreState, SnakeFoodState, \
	SnakeFoodDimensionState, SnakeFoodScoreState, SnakeFoodDimensionScoreState, DirectionalState, \
	DirectionalDimensionState, DirectionalScoreState, DirectionalDimensionScoreState, DirectionalDistanceState, \
	DirectionalDistanceDimensionState, DirectionalDistanceScoreState, DirectionalDistanceDimensionScoreState


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
