from PyQt5.QtWidgets import QApplication

from src.snake.gui import Window
from src.snake.parameters import SnakeParameters
from src.snake.agent import SnakePlayer
from src.snake.environment import SnakeEnvironment


def main():
	params = SnakeParameters()
	env = SnakeEnvironment(params)
	player = SnakePlayer()

	app = QApplication(["SnakeBot"])
	window = Window(env, params)
	window.show()
	app.exec_()


if __name__ == "__main__":
	main()
