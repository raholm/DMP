from src.snake.snake import SnakeDirection


class SnakeParameters(object):
	def __init__(self):
		# Game Related
		self.update_rate = 100

		# Board Related
		self.rows = 32
		self.cols = 32
		self.cell_size = 16

		# Snake Related
		self.initial_snake_size = 4
		self.initial_snake_position = (0, 0)
		self.initial_snake_direction = SnakeDirection.East
		self.tail_size_increase = 4