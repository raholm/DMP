from collections import deque
from enum import Enum

import random
from pygame import rect

import pygame


class SnakeParameters(object):
	def __init__(self):
		# Board Related
		self.rows = 32
		self.cols = 32
		self.cell_size = 16

		# Snake Related
		self.initial_snake_size = 4
		self.initial_snake_position = (0, 0)
		self.initial_snake_direction = Direction.East
		self.tail_size_increase = 4


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


class Player(object):
	def get_action(self, events):
		for event in events:
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_UP:
					self.next_dir.appendleft(Direction.North)
				elif event.key == pygame.K_DOWN:
					self.next_dir.appendleft(Direction.South)
				elif event.key == pygame.K_RIGHT:
					self.next_dir.appendleft(Direction.East)
				elif event.key == pygame.K_LEFT:
					self.next_dir.appendleft(Direction.West)


class Snake(object):
	def __init__(self, direction=Direction.East, head=(0, 0)):
		self.tail_size = 4
		self.direction = direction
		self.body = deque()
		self.body.append(head)
		self.next_dir = deque()

	@property
	def head(self):
		return self.body[0]

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

		# head = self.deque.pop()
		# self.deque.append(head)
		next_move = self.head

		if next_dir == Direction.North:
			if self.direction != Direction.South:
				next_move = (self.head[0] - 1, self.head[1])
				self.direction = next_dir
			else:
				next_move = (self.head[0] + 1, self.head[1])
		elif next_dir == Direction.South:
			if self.direction != Direction.North:
				next_move = (self.head[0] + 1, self.head[1])
				self.direction = next_dir
			else:
				next_move = (self.head[0] - 1, self.head[1])
		elif next_dir == Direction.West:
			if self.direction != Direction.East:
				next_move = (self.head[0], self.head[1] - 1)
				self.direction = next_dir
			else:
				next_move = (self.head[0], self.head[1] + 1)
		elif next_dir == Direction.East:
			if self.direction != Direction.West:
				next_move = (self.head[0], self.head[1] + 1)
				self.direction = next_dir
			else:
				next_move = (self.head[0], self.head[1] - 1)

		self.body.append(next_move)

		if len(self.body) > self.tail_size:
			self.body.popleft()


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


class SnakeEnvironment(object):
	def __init__(self, params):
		self.snake = Snake(head=params.initial_snake_position,
						   direction=params.initial_snake_direction)
		self.food = None
		self.board = Board(params.rows, params.cols)
		self.food_count = 0

		self.__update_board()

	def update(self, action):
		self.snake.update(action)

		if self.__is_dead():
			return self.food_count

		if self.__got_food():
			self.food_count += 1
			self.__update_food()

		self.__update_board()

	def __is_dead(self):
		# Snake is outside board
		if (self.snake.head[0] < 0 or self.snake.head[0] >= self.board.rows) or \
				(self.snake.head[1] < 0 or self.snake.head[1] >= self.board.cols):
			return True
		# Snake is colliding with itself
		if self.board[self.snake.head[0], self.snake.head[1]] == GridType.Snake:
			return True
		return False

	def __got_food(self):
		return self.board[self.snake.head] == GridType.Food

	def __update_food(self):
		while True:
			food = random.randrange(self.board.rows), random.randrange(self.board.cols)
			if not (self.board[food[0], food[1]] == GridType.Snake or
							self.board[food[0], food[1]] == GridType.Food):
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


class SnakeRenderer(object):
	def __init__(self, env, params):
		self.env = env
		self.params = params
		self.screen = None

	def init(self):
		pygame.init()
		self.screen = pygame.display.set_mode([self.params.rows * self.params.cell_size,
											   self.params.cols * self.params.cell_size])
		pygame.display.set_caption("Snake")
		pygame.draw.rect(self.screen, Color.Black.value, pygame.Rect(50, 50, 10, 10))

	def quit(self):
		pygame.quit()

	def render(self):
		self.screen.fill(Color.Black.value)

		rect = pygame.Rect(0, 0, self.params.cell_size, self.params.cell_size)

		for row in range(self.params.rows):
			for col in range(self.params.cols):
				moved_rect = rect.move(row * self.params.cell_size, col * self.params.cell_size)

				if self.env.board[row, col] == GridType.Snake:
					color = Color.Green.value
				elif self.env.board[row, col] == GridType.Food:
					color = Color.Red.value
				else:
					color = Color.Black.value

				pygame.draw.rect(self.screen, color, moved_rect)

		pygame.display.update()


# Return 0 to exit the program, 1 for a one-player game
def menu(screen):
	font = pygame.font.Font(None, 30)
	menu_message1 = font.render("Press enter to start.", True, Color.White.value)

	screen.fill(Color.Black.value)
	screen.blit(menu_message1, (32, 32))
	pygame.display.update()

	while True:
		done = False
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				done = True
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_RETURN:
					return 1
				if event.key == pygame.K_l:
					return 2
		if done:
			break

	if done:
		pygame.quit()
		return 0


def quit(screen):
	return False


def move(snake):
	if len(snake.next_dir) != 0:
		next_dir = snake.next_dir.pop()
	else:
		next_dir = snake.direction

	head = snake.deque.pop()
	snake.deque.append(head)
	next_move = head

	if next_dir == Direction.North:
		if snake.direction != Direction.South:
			next_move = (head[0] - 1, head[1])
			snake.direction = next_dir
		else:
			next_move = (head[0] + 1, head[1])
	elif next_dir == Direction.South:
		if snake.direction != Direction.North:
			next_move = (head[0] + 1, head[1])
			snake.direction = next_dir
		else:
			next_move = (head[0] - 1, head[1])
	elif next_dir == Direction.West:
		if snake.direction != Direction.East:
			next_move = (head[0], head[1] - 1)
			snake.direction = next_dir
		else:
			next_move = (head[0], head[1] + 1)
	elif next_dir == Direction.East:
		if snake.direction != Direction.West:
			next_move = (head[0], head[1] + 1)
			snake.direction = next_dir
		else:
			next_move = (head[0], head[1] - 1)
	return next_move


# Return false to quit program, true to go to
# gameover screen
def one_player(screen):
	clock = pygame.time.Clock()
	spots = make_board()

	snake = Snake()
	# Board set up
	spots[0][0] = GridType.Snake
	food = find_food(spots)

	while True:
		clock.tick(15)
		# Event processing
		done = False
		events = pygame.event.get()
		for event in events:
			if event.type == pygame.QUIT:
				print("Quit given")
				done = True
				break
		if done:
			return False

		snake.populate_next_dir(events)

		# Game logic
		next_head = move(snake)
		if end_condition(spots, next_head):
			return snake.tail_size

		if is_food(spots, next_head):
			snake.tail_size += 4
			food = find_food(spots)

		snake.body.append(next_head)

		if len(snake.body) > snake.tail_size:
			snake.body.popleft()

		# Draw code
		screen.fill(Color.Black.value)  # makes screen Color.Black

		spots = update_board(screen, [snake], food)

		pygame.display.update()


def game_over(screen, eaten):
	message1 = "You ate %d foods" % eaten
	message2 = "Press enter to play again, esc to quit."
	game_over_message1 = pygame.font.Font(None, 30).render(message1, True, Color.Black.value)
	game_over_message2 = pygame.font.Font(None, 30).render(message2, True, Color.Black.value)

	overlay = pygame.Surface((BOARD_LENGTH * OFFSET, BOARD_LENGTH * OFFSET))
	overlay.fill((84, 84, 84))
	overlay.set_alpha(150)
	screen.blit(overlay, (0, 0))

	screen.blit(game_over_message1, (35, 35))
	screen.blit(game_over_message2, (65, 65))
	game_over_message1 = pygame.font.Font(None, 30).render(message1, True, Color.White.value)
	game_over_message2 = pygame.font.Font(None, 30).render(message2, True, Color.White.value)
	screen.blit(game_over_message1, (32, 32))
	screen.blit(game_over_message2, (62, 62))

	pygame.display.update()

	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					return False
				if event.key == pygame.K_RETURN:
					return True


def leaderboard(screen):
	font = pygame.font.Font(None, 30)
	screen.fill(Color.Black.value)

	try:
		with open("leaderboard.txt") as f:
			lines = f.readlines()
			titlemessage = font.render("Leaderboard", True, Color.White.value)
			screen.blit(titlemessage, (32, 32))
			dist = 64
			for line in lines:
				delimited = line.split(",")
				delimited[1] = delimited[1].strip()
				message = "{0[0]:.<10}{0[1]:.>10}".format(delimited)
				rendered_message = font.render(message, True, Color.White.value)
				screen.blit(rendered_message, (32, dist))
				dist += 32
	except IOError:
		message = "Nothing on the leaderboard yet."
		rendered_message = font.render(message, True, Color.White.value)
		screen.blit(rendered_message, (32, 32))

	pygame.display.update()

	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					return False
				if event.key == pygame.K_RETURN:
					return True


# def main():
# 	pygame.init()
# 	screen = pygame.display.set_mode([BOARD_LENGTH * OFFSET, BOARD_LENGTH * OFFSET])
# 	pygame.display.set_caption("Snake")
# 	pygame.draw.rect(screen, pygame.Color(255, 255, 255, 255), pygame.Rect(50, 50, 10, 10))
# 	first = True
# 	playing = True
# 	while playing:
# 		if first or pick == 3:
# 			pick = menu(screen)
#
# 		options = {0: quit,
# 				   1: one_player,
# 				   2: leaderboard}
# 		now = options[pick](screen)
# 		if not now:
# 			break
# 		elif pick == 1 or pick == 2:
# 			eaten = now / 4 - 1
# 			playing = game_over(screen, eaten)
# 			first = False
#
# 	pygame.quit()

def main():
	params = SnakeParameters()
	env = SnakeEnvironment(params)
	renderer = SnakeRenderer(env, params)

	renderer.init()

	renderer.render()

	renderer.quit()


if __name__ == "__main__":
	main()
