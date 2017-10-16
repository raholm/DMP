from src.core.discount_factor import StaticDiscountFactor
from src.core.value_function import DictActionValueFunction
from src.snake.reward import DefaultSnakeReward
from src.snake.snake import SnakeDirection
from src.snake.state import DistanceState, SnakeAndFoodState, WholeState, SnakeHeadTailAndFoodState, \
	SnakeHeadAndFoodState


class SnakeParameters(object):
	def __init__(self):
		# Game Related
		self.update_rate = 100

		# Board Related
		self.rows = 5
		self.cols = 5
		self.cell_size = 48

		# Snake Related
		self.initial_snake_size = 1
		self.initial_snake_position = (0, 0)
		self.initial_snake_direction = SnakeDirection.East
		self.tail_size_increase = 1

		self.state = SnakeAndFoodState
		self.reward = DefaultSnakeReward

		# Learning Related
		self.discount_factor = StaticDiscountFactor(0.85)
		self.learning_rate = 0.15
		self.epsilon = 0.2
		self.value_function = DictActionValueFunction(0)
		self.train_episodes = 10000000
