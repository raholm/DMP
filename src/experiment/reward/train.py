from timeit import default_timer as timer

import numpy as np

from src.experiment.train import train_and_store_model
from src.snake.reward import *


def train_reward_models(params):
	rewards = [PosTravelPosScore, NegTravelPosScore,
			   ZeroTravelPosScore, PosTravelNegScore,
			   NegTravelNegScore, ZeroTravelNegScore,
			   PosTravelZeroScore, NegTravelZeroScore,
			   ZeroTravelZeroScore,
			   NegDistancePosBodySize,
			   NegDistanceNegTimeStepPosBodySize,
			   NegDistanceNegSelfCollisionPosBodySize,
			   NegDistanceNegBorderCollisionPosBodySize]

	for reward in rewards:
		params.model_params.reward = reward
		train_and_store_model(params)


def train_models(params):
	np.random.seed(params.seed)

	start = timer()

	train_reward_models(params)

	print("Elapsed time:", timer() - start)
