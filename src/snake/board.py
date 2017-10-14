from enum import Enum


class SnakeCellType(Enum):
	Empty = 0
	Snake = 1
	Food = 2

	def __str__(self):
		return str(self.value)


class SnakeBoard(object):
	def __init__(self, rows, cols):
		self.grid = [[SnakeCellType.Empty] * cols for _ in range(rows)]

	@property
	def rows(self):
		return len(self.grid)

	@property
	def cols(self):
		return len(self.grid[0])

	def __setitem__(self, index, value):
		self.grid[index[0]][index[1]] = value

	def __getitem__(self, index):
		return self.grid[index[0]][index[1]]
