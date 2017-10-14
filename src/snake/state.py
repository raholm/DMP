from src.core.state import State
from src.snake.board import SnakeCellType
from src.util.math import manhattan


class SnakeState(State):
	def _key(self):
		raise NotImplementedError

	def __eq__(self, other):
		return other and other.__key() == self._key()

	def __hash__(self):
		return hash(self._key())

	def __repr__(self):
		return str(self._key())


class WholeState(SnakeState):
	def __init__(self, env):
		self.board = "".join(str(cell)
							 for row in env.board.grid
							 for cell in row)

	def _key(self):
		return self.board


class SnakeAndFoodState(SnakeState):
	def __init__(self, env):
		self.data = "".join(str((row_num, col_num))
							for row_num, row in enumerate(env.board.grid)
							for col_num, cell in enumerate(row)
							if not cell == SnakeCellType.Empty)

	def _key(self):
		return self.data


class DistanceState(SnakeState):
	def __init__(self, env):
		self.snake_head = tuple(env.snake.head)
		self.snake_tail = tuple(env.snake.tail)
		self.distance = manhattan(self.snake_head, env.food)

	def _key(self):
		return self.snake_head, self.snake_tail, self.distance
