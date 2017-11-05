from timeit import default_timer as timer

import numpy as np

from src.experiment.train import train_and_store_model
from src.snake.reward import PositiveTravelPositiveFood, NegativeTravelPositiveFood, ZeroTravelPositiveFood, \
	NegativeTravelNegativeFood, PositiveTravelNegativeFood, ZeroTravelNegativeFood, PositiveTravelZeroFood, \
	NegativeTravelZeroFood, ZeroTravelZeroFood


def train_reward_models(params):
	rewards = [PositiveTravelPositiveFood, NegativeTravelPositiveFood,
			   ZeroTravelPositiveFood, PositiveTravelNegativeFood,
			   NegativeTravelNegativeFood, ZeroTravelNegativeFood,
			   PositiveTravelZeroFood, NegativeTravelZeroFood,
			   ZeroTravelZeroFood]

	for reward in rewards:
		params.model_params.reward = reward
		train_and_store_model(params)


def train_models(params):
	np.random.seed(params.seed)

	start = timer()

	train_reward_models(params)

	print("Elapsed time:", timer() - start)
