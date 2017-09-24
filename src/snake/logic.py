import random

from collections import deque
from enum import Enum

from src.core.action import Action
from src.core.agent import Agent
from src.core.environment import Environment


class SnakeGridType(Enum):
	Empty = 0
	Snake = 1
	Food = 2


class SnakeDirection(Enum):
	North = 0
	South = 1
	West = 2
	East = 3


class SnakeAction(Action, Enum):
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
		self.initial_snake_direction = SnakeDirection.East
		self.tail_size_increase = 4


class Player(Agent):
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
		if action == SnakeAction.North:
			self.next_dir.appendleft(SnakeDirection.North)
		elif action == SnakeAction.South:
			self.next_dir.appendleft(SnakeDirection.South)
		elif action == SnakeAction.East:
			self.next_dir.appendleft(SnakeDirection.East)
		elif action == SnakeAction.West:
			self.next_dir.appendleft(SnakeDirection.West)

	def __update_body(self):
		if len(self.next_dir) != 0:
			next_dir = self.next_dir.pop()
		else:
			next_dir = self.direction

		next_move = None

		if next_dir == SnakeDirection.North:
			if self.direction != SnakeDirection.South:
				next_move = (self.head[0], self.head[1] - 1)
				self.direction = next_dir
			else:
				next_move = (self.head[0], self.head[1] + 1)
		elif next_dir == SnakeDirection.South:
			if self.direction != SnakeDirection.North:
				next_move = (self.head[0], self.head[1] + 1)
				self.direction = next_dir
			else:
				next_move = (self.head[0], self.head[1] - 1)
		elif next_dir == SnakeDirection.West:
			if self.direction != SnakeDirection.East:
				next_move = (self.head[0] - 1, self.head[1])
				self.direction = next_dir
			else:
				next_move = (self.head[0] + 1, self.head[1])
		elif next_dir == SnakeDirection.East:
			if self.direction != SnakeDirection.West:
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


class SnakeBoard(object):
	def __init__(self, rows, cols):
		self.board = [[SnakeGridType.Empty] * cols for _ in range(rows)]

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
		self.board = SnakeBoard(params.rows, params.cols)
		self.food_count = 0

		self.__update_food_position()
		self.__update_board()

	def update(self, action):
		self.snake.update(action)

		if self.__snake_is_dead():
			return False

		if self.__snake_got_food():
			self.snake.increase_size()
			self.food_count += 1
			self.__update_food_position()

		self.__update_board()

		return True

	def __snake_is_dead(self):
		# Snake is outside board
		if (self.snake.head[0] < 0 or self.snake.head[0] >= self.board.rows) or \
				(self.snake.head[1] < 0 or self.snake.head[1] >= self.board.cols):
			return True

		# Snake is colliding with itself
		if self.board[self.snake.head] == SnakeGridType.Snake:
			return True

		return False

	def __snake_got_food(self):
		return self.board[self.snake.head] == SnakeGridType.Food

	def __update_food_position(self):
		while True:
			food = random.randrange(self.board.rows), random.randrange(self.board.cols)
			if not (self.board[food] == SnakeGridType.Snake or self.board[food] == SnakeGridType.Food):
				break

		self.food = food

	def __update_board(self):
		for row in range(self.board.rows):
			for col in range(self.board.cols):
				self.board[row, col] = SnakeGridType.Empty

		if self.food is not None:
			self.board[self.food] = SnakeGridType.Food

		for body_part in self.snake.body:
			self.board[body_part] = SnakeGridType.Snake
