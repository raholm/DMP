from src.core.discount_factor import StaticDiscountFactor
from src.core.learning_rate import StaticLearningRate
from src.snake.parameters import SnakeParameters
from src.snake.reward import NegTravelPosScore
from src.snake.state import SnakeFoodScoreState


def get_parameters_seed():
	return 234


def get_snake_parameters():
	params = SnakeParameters()
	params.state = SnakeFoodScoreState
	params.reward = NegTravelPosScore
	params.train_episodes = 1000000

	learning_rates = [0.15, 0.5, 0.85]
	discount_factors = [0.85, 0.95, 1]

	for learning_rate in learning_rates:
		for discount_factor in discount_factors:
			params.learning_rate = StaticLearningRate(learning_rate)
			params.discount_factor = StaticDiscountFactor(discount_factor)
			yield params
