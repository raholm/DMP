from src.snake.reward import NegTravelPosScore
from src.snake.snake import SnakeDirection
from src.snake.state import SnakeFoodScoreState


class SnakeParameters(object):
	def __init__(self):
		# Game Related
		self.update_rate = 250

		# Board Related
		self.rows = 5
		self.cols = 5
		self.cell_size = 48

		# Snake Related
		self.initial_snake_size = 1
		self.initial_snake_position = (2, 2)
		self.initial_snake_direction = SnakeDirection.East
		self.tail_size_increase = 1

		# Learning Related
		self.state = SnakeFoodScoreState
		self.reward = NegTravelPosScore
