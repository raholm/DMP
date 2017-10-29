from src.core.state import State
from src.snake.board import SnakeCellType
from src.snake.snake import SnakeDirection
from src.util.math import manhattan


class SnakeState(State):
	def key(self):
		raise NotImplementedError

	def __eq__(self, other):
		return other and other.key() == self.key()

	def __hash__(self):
		return hash(self.key())

	def __repr__(self):
		return str(self.key())


class BoardState(SnakeState):
	def __init__(self, env):
		self.board = "".join(str(cell)
							 for row in env.board.grid
							 for cell in row) + " " + str(env.score)

	def key(self):
		return self.board


class SnakeAndFoodWithScoreState(SnakeState):
	def __init__(self, env):
		self.data = (tuple((row, col)
						   for row in range(env.board.rows)
						   for col in range(env.board.cols)
						   if not env.board[row, col] == SnakeCellType.Empty),
					 env.score)

	def key(self):
		return self.data


class SnakeAndFoodWithoutScoreState(SnakeState):
	def __init__(self, env):
		self.data = tuple((row, col)
						  for row in range(env.board.rows)
						  for col in range(env.board.cols)
						  if not env.board[row, col] == SnakeCellType.Empty)

	def key(self):
		return self.data


class DistanceState(SnakeState):
	def __init__(self, env):
		self.snake_head = tuple(env.snake.head)
		self.snake_tail = tuple(env.snake.tail)
		self.distance = manhattan(self.snake_head, env.food)
		self.score = env.score

	def key(self):
		return self.snake_head, self.snake_tail, self.distance, self.score


class DirectionalDistanceState(SnakeState):
	def __init__(self, env):
		self.distance = manhattan(env.snake.head, env.food)
		self.direction = self.__get_direction(env)
		self.score = env.score

	def key(self):
		return self.distance, self.direction, self.score

	@staticmethod
	def __get_direction(env):
		snake_x, snake_y = env.snake.head
		snake_direction = env.snake.direction
		food_x, food_y = env.food

		direction = snake_direction

		if snake_x == food_x:
			if snake_y < food_y:
				direction = SnakeDirection.South
			elif snake_y > food_y:
				direction = SnakeDirection.North
		elif snake_x < food_x:
			direction = SnakeDirection.East
		else:
			direction = SnakeDirection.West

		return direction
