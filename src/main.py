from PyQt5.QtWidgets import QApplication

from src.snake.gui import Window
from src.snake.parameters import SnakeParameters
from src.snake.agent import SnakePlayer
from src.snake.environment import SnakeEnvironment
from src.snake.state import WholeState, DistanceState, SnakeAndFoodState


def test_state():
	params = SnakeParameters()
	env = SnakeEnvironment(params)

	state = WholeState(env)
	print(state)
	print(hash(state))

	state = DistanceState(env)
	print(state)
	print(hash(state))

	state = SnakeAndFoodState(env)
	print(state)
	print(hash(state))


def main():
	params = SnakeParameters()
	env = SnakeEnvironment(params)
	player = SnakePlayer()

	app = QApplication(["SnakeBot"])
	window = Window(env, params)
	window.show()
	app.exec_()


if __name__ == "__main__":
	test_state()
