import sys

from PyQt5.QtCore import Qt, pyqtSlot, QBasicTimer
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtWidgets import *

from src.snake.logic import SnakeParameters, SnakeEnvironment, Player, GridType, Color, Action


class GameWidget(QWidget):
	def __init__(self, env, params, parent=None):
		super().__init__(parent)
		self.env = env
		self.params = params

	def paintEvent(self, event):
		painter = QPainter()
		painter.begin(self)

		for row in range(self.params.rows):
			for col in range(self.params.cols):
				width = self.params.cell_size
				height = self.params.cell_size
				x = row * width
				y = col * height

				if self.env.board[row, col] == GridType.Snake:
					color = Color.Green.value
				elif self.env.board[row, col] == GridType.Food:
					color = Color.Red.value
				else:
					color = Color.Black.value

				painter.setBrush(QColor(color[0], color[1], color[2], 255))
				painter.drawRect(x, y, width, height)

		painter.end()


class AgentWidget(QWidget):
	pass


class PlayerWidget(AgentWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.action = Action.East

	def keyPressEvent(self, event):
		if event.key() == Qt.Key_Up:
			self.action = Action.North
		elif event.key() == Qt.Key_Down:
			self.action = Action.South
		elif event.key() == Qt.Key_Left:
			self.action = Action.West
		elif event.key() == Qt.Key_Right:
			self.action = Action.East


class CreateWidget(QWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.layout = QVBoxLayout()

		self.algorithm_combo_box = QComboBox(self)
		self.algorithm_combo_box.addItem("Value Iteration")
		self.algorithm_combo_box.addItem("Policy Iteration")
		self.algorithm_combo_box.addItem("Monte Carlo")
		self.algorithm_combo_box.addItem("Sarsa")
		self.algorithm_combo_box.addItem("TD")
		self.algorithm_combo_box.addItem("Q-Learning")
		self.algorithm_combo_box.setMinimumWidth(150)

		self.layout.addWidget(self.algorithm_combo_box)

		self.button_layout = QHBoxLayout()

		self.create_button = QPushButton("Create")
		self.cancel_button = QPushButton("Cancel")
		self.cancel_button.clicked.connect(self.parent().close)

		self.button_layout.addWidget(self.create_button)
		self.button_layout.addWidget(self.cancel_button)

		self.layout.addLayout(self.button_layout)

		self.setLayout(self.layout)


class CreateWindow(QMainWindow):
	def __init__(self, parent=None):
		super().__init__(parent)
		# self.setGeometry(50, 50, 500, 300)
		self.setWindowTitle("Create Agent")
		self.create_widget = CreateWidget(self)
		self.setCentralWidget(self.create_widget)


class Window(QMainWindow):
	def __init__(self, env, params):
		super().__init__()
		self.setGeometry(50, 50, params.rows * params.cell_size, params.cols * params.cell_size)
		self.setWindowTitle("SnakeBot!")

		self.game_widget = GameWidget(env, params, self)
		self.setCentralWidget(self.game_widget)

		self.create_widget = CreateWindow(self)
		self.agent_widget = PlayerWidget(self)

		self.__create_menu()

		self.env = env
		self.params = params

		self.timer = QBasicTimer()
		self.timer.start(params.update_rate, self)
		self.update()

	def keyPressEvent(self, event):
		self.agent_widget.keyPressEvent(event)

	def __create_menu(self):
		create_action = QAction("&Create", self)
		create_action.setShortcut("Ctrl+C")
		create_action.setStatusTip('Create Agent')
		create_action.triggered.connect(self.on_push_create)

		save_action = QAction("&Save", self)
		save_action.setShortcut("Ctrl+S")
		save_action.setStatusTip('Save Agent')
		save_action.triggered.connect(self.close_application)

		load_action = QAction("&Load", self)
		load_action.setShortcut("Ctrl+L")
		load_action.setStatusTip('Load Agent')
		load_action.triggered.connect(self.close_application)

		quit_action = QAction("&Quit", self)
		quit_action.setShortcut("Ctrl+Q")
		quit_action.setStatusTip('Quit Agent')
		quit_action.triggered.connect(self.close_application)

		main_menu = self.menuBar()
		file_menu = main_menu.addMenu('&File')
		file_menu.addAction(create_action)
		file_menu.addAction(save_action)
		file_menu.addAction(load_action)
		file_menu.addAction(quit_action)

	@pyqtSlot()
	def on_push_create(self):
		self.create_widget.show()

	def close_application(self):
		sys.exit()

	def timerEvent(self, event):
		if event.timerId() == self.timer.timerId():
			self.env.update(self.agent_widget.action)
			self.repaint()


if __name__ == '__main__':
	params = SnakeParameters()
	env = SnakeEnvironment(params)
	player = Player()

	app = QApplication(["SnakeBot"])
	window = Window(env, params)
	window.show()
	app.exec_()
