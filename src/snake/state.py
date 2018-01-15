from collections import defaultdict

from src.core.state import State
from src.snake.board import SnakeCellType
from src.snake.snake import SnakeDirection
from src.util.math import manhattan


class SnakeState(State):
	def __init__(self, env):
		self.key = tuple(cls(env).data
						 for cls in self.data_classes)

	@property
	def data_classes(self):
		raise NotImplementedError

	def __eq__(self, other):
		return other and other.key == self.key

	def __hash__(self):
		return hash(self.key)

	def __repr__(self):
		return str(self.key)


class DimensionData(object):
	def __init__(self, env):
		self.data = env.board.rows, env.board.cols


class ScoreData(object):
	def __init__(self, env):
		self.data = env.score


class BoardData(object):
	def __init__(self, env):
		self.data = "".join(str(cell)
							for row in env.board.grid
							for cell in row)


class SnakeFoodData(object):
	def __init__(self, env):
		self.data = self.__get_snake(env), self.__get_food(env)

	@staticmethod
	def __get_snake(env):
		return tuple((row, col)
					 for row in range(env.board.rows)
					 for col in range(env.board.cols)
					 if env.board[row, col] == SnakeCellType.Snake)

	@staticmethod
	def __get_food(env):
		return tuple((row, col)
					 for row in range(env.board.rows)
					 for col in range(env.board.cols)
					 if env.board[row, col] == SnakeCellType.Food)[0]


class DistanceData(object):
	def __init__(self, env):
		self.data = manhattan(env.snake.head, env.food)


class DirectionalData(object):
	def __init__(self, env):
		self.data = self.__get_direction(env)

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


class ShortestPathData(object):
	def __init__(self, env):
		self.data = self.__get_shortest_path(env)

	def __get_shortest_path(self, env):
		if env.episode_is_done():
			return ()

		graph = self.__create_adjacency_list(env)

		start = env.snake.head
		goal = env.food

		explored = set()
		queue = [[start]]

		if start == goal:
			return start

		while queue:
			path = queue.pop(0)
			node = path[-1]

			if node not in explored:
				neighbours = graph[node]

				for neighbour in neighbours:
					new_path = list(path)
					new_path.append(neighbour)
					queue.append(new_path)

					if neighbour == goal:
						return tuple(new_path)

				explored.add(node)

		raise ValueError("Could not find a path between %s and %s" % (start, goal))

	def __create_adjacency_list(self, env):
		def add(adj_list, a, b):
			adj_list[a].add(b)
			adj_list[b].add(a)

		adj_list = defaultdict(set)

		for row in range(env.rows):
			for col in range(env.cols):
				if col < env.cols - 1:
					add(adj_list, (row, col), (row, col + 1))
				if row < env.rows - 1:
					for x in range(max(0, col - 1), min(env.cols, env.cols + 2)):
						add(adj_list, (row, col), (row + 1, col))

		return adj_list


class BoardState(SnakeState):
	@property
	def data_classes(self):
		return [BoardData]


class BoardDimensionState(SnakeState):
	@property
	def data_classes(self):
		return [BoardData, DimensionData]


class BoardScoreState(SnakeState):
	@property
	def data_classes(self):
		return [BoardData, ScoreData]


class BoardDimensionScoreState(SnakeState):
	@property
	def data_classes(self):
		return [BoardData, DimensionData, ScoreData]


class SnakeFoodState(SnakeState):
	@property
	def data_classes(self):
		return [SnakeFoodData]


class SnakeFoodDimensionState(SnakeState):
	@property
	def data_classes(self):
		return [SnakeFoodData, DimensionData]


class SnakeFoodScoreState(SnakeState):
	@property
	def data_classes(self):
		return [SnakeFoodData, ScoreData]


class SnakeFoodDimensionScoreState(SnakeState):
	@property
	def data_classes(self):
		return [SnakeFoodData, DimensionData, ScoreData]


class DirectionalState(SnakeState):
	@property
	def data_classes(self):
		return [DirectionalData]


class DirectionalDimensionState(SnakeState):
	@property
	def data_classes(self):
		return [DirectionalData, DimensionData]


class DirectionalScoreState(SnakeState):
	@property
	def data_classes(self):
		return [DirectionalData, ScoreData]


class DirectionalDimensionScoreState(SnakeState):
	@property
	def data_classes(self):
		return [DirectionalData, DimensionData, ScoreData]


class DirectionalDistanceState(SnakeState):
	@property
	def data_classes(self):
		return [DirectionalData, DistanceData]


class DirectionalDistanceDimensionState(SnakeState):
	@property
	def data_classes(self):
		return [DirectionalData, DistanceData, DimensionData]


class DirectionalDistanceScoreState(SnakeState):
	@property
	def data_classes(self):
		return [DirectionalData, DistanceData, ScoreData]


class DirectionalDistanceDimensionScoreState(SnakeState):
	@property
	def data_classes(self):
		return [DirectionalData, DistanceData, DimensionData, ScoreData]


class ShortestPathScoreState(SnakeState):
	@property
	def data_classes(self):
		return [ShortestPathData, ScoreData]
