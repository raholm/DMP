import numpy as np

from src.core.environment import Environment
from src.snake.action import SnakeAction
from src.snake.board import SnakeBoard, SnakeCellType
from src.snake.snake import Snake


class SnakeEnvironment(Environment):
	def __init__(self, params):
		self.params = params
		self.board = SnakeBoard(params.rows, params.cols)
		self.snake = None
		self.food = None
		self.food_count = None
		self.time_step = None
		self.episode_running = True
		self.death_from_self_collision = None

		self.start_new_episode()

	def start_new_episode(self):
		self.snake = self.__create_snake()
		self.food = self.__create_food()
		self.food_count = 0
		self.time_step = 0
		self.episode_running = True
		self.death_from_self_collision = False

		self.__update_board()

		return self.params.state(self)

	def episode_is_done(self):
		return not self.episode_running

	def get_valid_actions(self, state):
		return [SnakeAction.South, SnakeAction.North, SnakeAction.West, SnakeAction.East]

	def step(self, action):
		self.time_step += 1

		old_state = self.params.state(self)

		self.snake.update(action)

		if self.__snake_is_dead():
			self.episode_running = False

		if self.episode_running:
			if self.__snake_got_food():
				self.snake.increase_size()
				self.food_count += 1
				self.food = self.__create_food()

			self.__update_board()

		new_state = self.params.state(self)
		reward = self.params.reward(self, old_state, action, new_state)
		return new_state, reward

	def run(self, agent, n_episodes):
		scores = [None] * n_episodes

		for iteration in range(n_episodes):
			state = self.start_new_episode()

			while not self.episode_is_done():
				action = agent.get_action(state)
				state, _ = self.step(action)

			scores[iteration] = self.score

		return scores

	@property
	def score(self):
		return self.food_count * self.game_score_coef

	@property
	def game_score_coef(self):
		return 100

	@property
	def rows(self):
		return self.board.rows

	@property
	def cols(self):
		return self.board.cols

	def __snake_is_dead(self):
		# Snake is outside board
		if (self.snake.head[0] < 0 or self.snake.head[0] >= self.board.rows) or \
				(self.snake.head[1] < 0 or self.snake.head[1] >= self.board.cols):
			self.death_from_self_collision = False
			return True

		# Snake is colliding with itself
		if self.board[self.snake.head] == SnakeCellType.Snake:
			self.death_from_self_collision = True
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
			food = np.random.randint(self.board.rows), np.random.randint(self.board.cols)
			if not (self.board[food] == SnakeCellType.Snake or
							self.board[food] == SnakeCellType.Food or food == self.snake.head):
				break

		return food
