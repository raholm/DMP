import random

from src.core.environment import Environment
from src.snake.action import SnakeAction
from src.snake.snake import Snake
from src.snake.board import SnakeBoard, SnakeCellType


class SnakeEnvironment(Environment):
	def __init__(self, params):
		self.params = params
		self.board = SnakeBoard(params.rows, params.cols)
		self.snake = None
		self.food = None
		self.food_count = None

		self.start_new_episode()

	def start_new_episode(self):
		self.snake = self.__create_snake()
		self.food = self.__create_food()
		self.food_count = 0

		self.__update_board()

	def episode_is_done(self):
		return self.__snake_is_dead()

	def get_valid_actions(self, state):
		return [SnakeAction.South, SnakeAction.North, SnakeAction.West, SnakeAction.East]

	def step(self, action):
		self.snake.update(action)

		if self.episode_is_done():
			return False

		if self.__snake_got_food():
			self.snake.increase_size()
			self.food_count += 1
			self.food = self.__create_food()

		self.__update_board()

		return True

	def __snake_is_dead(self):
		# Snake is outside board
		if (self.snake.head[0] < 0 or self.snake.head[0] >= self.board.rows) or \
				(self.snake.head[1] < 0 or self.snake.head[1] >= self.board.cols):
			return True

		# Snake is colliding with itself
		if self.board[self.snake.head] == SnakeCellType.Snake:
			return True

		return False

	def __snake_got_food(self):
		return self.board[self.snake.head] == SnakeCellType.Food

	def __update_board(self):
		for row in range(self.board.rows):
			for col in range(self.board.cols):
				self.board[row, col] = SnakeCellType.Empty

		if self.food is not None:
			self.board[self.food] = SnakeCellType.Food

		for body_part in self.snake.body:
			self.board[body_part] = SnakeCellType.Snake

	def __create_snake(self):
		return Snake(size=self.params.initial_snake_size,
					 head=self.params.initial_snake_position,
					 direction=self.params.initial_snake_direction,
					 tail_increase=self.params.tail_size_increase)

	def __create_food(self):
		while True:
			food = random.randrange(self.board.rows), random.randrange(self.board.cols)
			if not (self.board[food] == SnakeCellType.Snake or
							self.board[food] == SnakeCellType.Food or food == self.snake.head):
				break

		return food
