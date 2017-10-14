from enum import Enum


class SnakeGridType(Enum):
	Empty = 0
	Snake = 1
	Food = 2


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
