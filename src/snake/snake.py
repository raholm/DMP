from collections import deque
from enum import Enum

from src.snake.action import SnakeAction


class SnakeDirection(Enum):
	North = 0
	South = 1
	West = 2
	East = 3


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
