from enum import Enum

from src.core.action import Action


class SnakeAction(Action, Enum):
	North = 0
	South = 1
	West = 2
	East = 3
	Quit = 4

	def __eq__(self, other):
		return other and other.value == self.value

	def __hash__(self):
		return hash(self.value)
