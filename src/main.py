from PyQt5.QtWidgets import QApplication

from src.snake.gui import Window
from src.snake.logic import SnakeParameters, SnakeEnvironment, Player


def main():
	params = SnakeParameters()
	env = SnakeEnvironment(params)
	player = Player()

	app = QApplication(["SnakeBot"])
	window = Window(env, params)
	window.show()
	app.exec_()


if __name__ == "__main__":
	main()
