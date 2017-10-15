from src.snake.reward import DefaultSnakeReward
from src.snake.snake import SnakeDirection
from src.snake.state import DistanceState, SnakeAndFoodState, WholeState, SnakeHeadAndFoodState


class SnakeParameters(object):
	def __init__(self):
		# Game Related
		self.update_rate = 100

		# Board Related
		self.rows = 9
		self.cols = 9
		self.cell_size = 48

		# Snake Related
		self.initial_snake_size = 1
		self.initial_snake_position = (0, 0)
		self.initial_snake_direction = SnakeDirection.East
		self.tail_size_increase = 1

		self.state = SnakeHeadAndFoodState
		self.reward = DefaultSnakeReward
