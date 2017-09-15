from collections import deque
from enum import Enum

import random
import pygame

BOARD_LENGTH = 32
OFFSET = 16


class Parameters(object):
	def __int__(self):
		self.board_length = 32
		self.offset = 16


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


def snake_color():
	return Color.Green.value


def food_color():
	return Color.Red.value


class Snake(object):
	def __init__(self, direction=Direction.East, point=(0, 0, snake_color())):
		self.tailmax = 4
		self.direction = direction
		self.deque = deque()
		self.deque.append(point)
		self.color = snake_color()
		self.nextDir = deque()

	def get_color(self):
		return self.color

	def populate_next_dir(self, events):
		for event in events:
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_UP:
					self.nextDir.appendleft(Direction.North)
				elif event.key == pygame.K_DOWN:
					self.nextDir.appendleft(Direction.South)
				elif event.key == pygame.K_RIGHT:
					self.nextDir.appendleft(Direction.East)
				elif event.key == pygame.K_LEFT:
					self.nextDir.appendleft(Direction.West)


def find_food(spots):
	while True:
		food = random.randrange(BOARD_LENGTH), random.randrange(BOARD_LENGTH)
		if not (spots[food[0]][food[1]] == GridType.Snake or spots[food[0]][food[1]] == GridType.Food):
			break
	return food


def end_condition(board, coord):
	if (coord[0] < 0 or
				coord[0] >= BOARD_LENGTH or
				coord[1] < 0 or
				coord[1] >= BOARD_LENGTH):
		return True
	if board[coord[0]][coord[1]] == GridType.Snake:
		return True
	return False


def make_board():
	return [[GridType.Empty] * BOARD_LENGTH for _ in range(BOARD_LENGTH)]


def update_board(screen, snakes, food):
	rect = pygame.Rect(0, 0, OFFSET, OFFSET)
	spots = [[0] * BOARD_LENGTH for _ in range(BOARD_LENGTH)]

	# Draw background
	num1 = 0
	num2 = 0
	for _ in spots:
		for _ in range(BOARD_LENGTH):
			temprect = rect.move(num1 * OFFSET, num2 * OFFSET)
			pygame.draw.rect(screen, Color.Black.value, temprect)
			num2 += 1
		num1 += 1

	# Draw food
	spots[food[0]][food[1]] = GridType.Food
	temprect = rect.move(food[1] * OFFSET, food[0] * OFFSET)
	pygame.draw.rect(screen, food_color(), temprect)

	# Draw snake
	for snake in snakes:
		for coord in snake.deque:
			spots[coord[0]][coord[1]] = GridType.Snake
			temprect = rect.move(coord[1] * OFFSET, coord[0] * OFFSET)
			pygame.draw.rect(screen, coord[2], temprect)

	return spots


def get_color(s):
	if s == "bk":
		return Color.Black.value
	elif s == "wh":
		return Color.White.value
	elif s == "rd":
		return Color.Red.value
	elif s == "bl":
		return Color.Blue.value
	elif s == "fo":
		return food_color()
	else:
		print("WHAT", s)
		return Color.Blue.value


def update_board_delta(screen, deltas):
	# accepts a queue of deltas in the form
	# [("d", 13, 30), ("a", 4, 6, "rd")]
	# valid colors: re, wh, bk, bl
	rect = pygame.Rect(0, 0, OFFSET, OFFSET)
	change_list = []
	delqueue = deque()
	addqueue = deque()
	while len(deltas) != 0:
		d = deltas.pop()
		change_list.append(pygame.Rect(d[1], d[2], OFFSET, OFFSET))
		if d[0] == "d":
			delqueue.append((d[1], d[2]))
		elif d[0] == "a":
			addqueue.append((d[1], d[2], get_color(d[3])))

	for d_coord in delqueue:
		temprect = rect.move(d_coord[1] * OFFSET, d_coord[0] * OFFSET)
		# TODO generalize background color
		pygame.draw.rect(screen, Color.Black.value, temprect)

	for a_coord in addqueue:
		temprect = rect.move(a_coord[1] * OFFSET, a_coord[0] * OFFSET)
		pygame.draw.rect(screen, a_coord[2], temprect)

	return change_list


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
	if len(snake.nextDir) != 0:
		next_dir = snake.nextDir.pop()
	else:
		next_dir = snake.direction

	head = snake.deque.pop()
	snake.deque.append(head)
	next_move = head

	if next_dir == Direction.North:
		if snake.direction != Direction.South:
			next_move = (head[0] - 1, head[1], snake.get_color())
			snake.direction = next_dir
		else:
			next_move = (head[0] + 1, head[1], snake.get_color())
	elif next_dir == Direction.South:
		if snake.direction != Direction.North:
			next_move = (head[0] + 1, head[1], snake.get_color())
			snake.direction = next_dir
		else:
			next_move = (head[0] - 1, head[1], snake.get_color())
	elif next_dir == Direction.West:
		if snake.direction != Direction.East:
			next_move = (head[0], head[1] - 1, snake.get_color())
			snake.direction = next_dir
		else:
			next_move = (head[0], head[1] + 1, snake.get_color())
	elif next_dir == Direction.East:
		if snake.direction != Direction.West:
			next_move = (head[0], head[1] + 1, snake.get_color())
			snake.direction = next_dir
		else:
			next_move = (head[0], head[1] - 1, snake.get_color())
	return next_move


def is_food(board, point):
	return board[point[0]][point[1]] == GridType.Food


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
			return snake.tailmax

		if is_food(spots, next_head):
			snake.tailmax += 4
			food = find_food(spots)

		snake.deque.append(next_head)

		if len(snake.deque) > snake.tailmax:
			snake.deque.popleft()

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


def main():
	pygame.init()
	screen = pygame.display.set_mode([BOARD_LENGTH * OFFSET, BOARD_LENGTH * OFFSET])
	pygame.display.set_caption("Snake")
	pygame.draw.rect(screen, pygame.Color(255, 255, 255, 255), pygame.Rect(50, 50, 10, 10))
	first = True
	playing = True
	while playing:
		if first or pick == 3:
			pick = menu(screen)

		options = {0: quit,
				   1: one_player,
				   2: leaderboard}
		now = options[pick](screen)
		if not now:
			break
		elif pick == 1 or pick == 2:
			eaten = now / 4 - 1
			playing = game_over(screen, eaten)
			first = False

	pygame.quit()


if __name__ == "__main__":
	main()
