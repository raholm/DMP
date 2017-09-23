from collections import deque
from enum import Enum

import random


class Color(Enum):
	White = (255, 255, 255)
	Black = (0, 0, 0)
	Red = (255, 0, 0)
	Green = (0, 255, 0)
	Blue = (0, 0, 255)


class GridType(Enum):
	Empty = 0
	Snake = 1
	Food = 2


class Direction(Enum):
	North = 0
	South = 1
	West = 2
	East = 3


class Action(Enum):
	North = 0
	South = 1
	West = 2
	East = 3
	Quit = 4


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
		self.initial_snake_direction = Direction.East
		self.tail_size_increase = 4


class Agent(object):
	pass


class Player(Agent):
	pass


class Environment(object):
	pass


class Snake(object):
	def __init__(self, size, direction, head, tail_increase):
		self.tail_size = size
		self.tail_increase = tail_increase
		self.direction = direction
		self.body = deque()
		self.body.append(head)
		self.next_dir = deque()

	@property
	def head(self):
		return self.body[-1]

	def update(self, action):
		self.__update_direction(action)
		self.__update_body()

	def __update_direction(self, action):
		if action == Action.North:
			self.next_dir.appendleft(Direction.North)
		elif action == Action.South:
			self.next_dir.appendleft(Direction.South)
		elif action == Action.East:
			self.next_dir.appendleft(Direction.East)
		elif action == Action.West:
			self.next_dir.appendleft(Direction.West)

	def __update_body(self):
		if len(self.next_dir) != 0:
			next_dir = self.next_dir.pop()
		else:
			next_dir = self.direction

		next_move = None

		if next_dir == Direction.North:
			if self.direction != Direction.South:
				next_move = (self.head[0], self.head[1] - 1)
				self.direction = next_dir
			else:
				next_move = (self.head[0], self.head[1] + 1)
		elif next_dir == Direction.South:
			if self.direction != Direction.North:
				next_move = (self.head[0], self.head[1] + 1)
				self.direction = next_dir
			else:
				next_move = (self.head[0], self.head[1] - 1)
		elif next_dir == Direction.West:
			if self.direction != Direction.East:
				next_move = (self.head[0] - 1, self.head[1])
				self.direction = next_dir
			else:
				next_move = (self.head[0] + 1, self.head[1])
		elif next_dir == Direction.East:
			if self.direction != Direction.West:
				next_move = (self.head[0] + 1, self.head[1])
				self.direction = next_dir
			else:
				next_move = (self.head[0] - 1, self.head[1])

		if next_move is not None:
			self.body.append(next_move)

		if len(self.body) > self.tail_size:
			self.body.popleft()

	def increase_size(self):
		self.tail_size += self.tail_increase


class Board(object):
	def __init__(self, rows, cols):
		self.board = [[GridType.Empty] * cols for _ in range(rows)]

	@property
	def rows(self):
		return len(self.board)

	@property
	def cols(self):
		return len(self.board[0])

	def __setitem__(self, index, value):
		self.board[index[0]][index[1]] = value

	def __getitem__(self, index):
		return self.board[index[0]][index[1]]


class SnakeEnvironment(Environment):
	def __init__(self, params):
		self.snake = Snake(size=params.initial_snake_size,
						   head=params.initial_snake_position,
						   direction=params.initial_snake_direction,
						   tail_increase=params.tail_size_increase)
		self.food = None
		self.board = Board(params.rows, params.cols)
		self.food_count = 0

		self.__update_food()
		self.__update_board()

	def update(self, action):
		self.snake.update(action)

		if self.__is_dead():
			return False

		if self.__got_food():
			self.snake.increase_size()
			self.food_count += 1
			self.__update_food()

		self.__update_board()

		return True

	def __is_dead(self):
		# Snake is outside board
		if (self.snake.head[0] < 0 or self.snake.head[0] >= self.board.rows) or \
				(self.snake.head[1] < 0 or self.snake.head[1] >= self.board.cols):
			return True

		# Snake is colliding with itself
		if self.board[self.snake.head] == GridType.Snake:
			return True

		return False

	def __got_food(self):
		return self.board[self.snake.head] == GridType.Food

	def __update_food(self):
		while True:
			food = random.randrange(self.board.rows), random.randrange(self.board.cols)
			if not (self.board[food] == GridType.Snake or self.board[food] == GridType.Food):
				break

		self.food = food

	def __update_board(self):
		for row in range(self.board.rows):
			for col in range(self.board.cols):
				self.board[row, col] = GridType.Empty

		if self.food is not None:
			self.board[self.food] = GridType.Food

		for body_part in self.snake.body:
			self.board[body_part] = GridType.Snake
