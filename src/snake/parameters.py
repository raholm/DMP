from src.core.discount_factor import StaticDiscountFactor
from src.core.learning_rate import StaticLearningRate
from src.core.value_function import DictActionValueFunction
from src.snake.reward import NegTravelPosScore
from src.snake.snake import SnakeDirection
from src.snake.state import SnakeFoodScoreState


class SnakeParameters(object):
	def __init__(self):
		# Game Related
		self.update_rate = 250

		# Board Related
		self.rows = 3
		self.cols = 3
		self.cell_size = 48

		# Snake Related
		self.initial_snake_size = 1
		self.initial_snake_position = (1, 1)
		self.initial_snake_direction = SnakeDirection.East
		self.tail_size_increase = 1

		# Learning Related
		self.state = SnakeFoodScoreState
		self.reward = NegTravelPosScore
