import importlib
import os
import sys

from PyQt5.QtCore import Qt, pyqtSlot, QBasicTimer
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtWidgets import *

from src.core.policy import EpsilonGreedyPolicy
from src.snake.action import SnakeAction
from src.snake.agent import SnakePlayer, SnakeAgent
from src.snake.board import SnakeCellType
from src.snake.environment import SnakeEnvironment
from src.snake.parameters import SnakeParameters
from src.util.color import Color
from src.util.io import read_model


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

				if self.env.board[row, col] == SnakeCellType.Snake:
					if self.env.snake.head == (row, col):
						color = Color.Orange.value
					else:
						color = Color.Green.value
				elif self.env.board[row, col] == SnakeCellType.Food:
					color = Color.Red.value
				else:
					color = Color.Black.value

				painter.setBrush(QColor(color[0], color[1], color[2], 255))
				painter.drawRect(x, y, width, height)

		painter.end()


class AgentWidget(QWidget):
	def __init__(self, agent, parent=None):
		super().__init__(parent)
		self.agent = agent

	def get_action(self, state):
		return self.agent.get_action(state)

	def keyPressEvent(self, event):
		if not isinstance(self.agent, SnakePlayer):
			return

		if event.key() == Qt.Key_Up:
			self.agent.action = SnakeAction.North
		elif event.key() == Qt.Key_Down:
			self.agent.action = SnakeAction.South
		elif event.key() == Qt.Key_Left:
			self.agent.action = SnakeAction.West
		elif event.key() == Qt.Key_Right:
			self.agent.action = SnakeAction.East


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
	def __init__(self, env, agent, params):
		super().__init__()
		self.setGeometry(50, 50,
						 params.rows * params.cell_size,
						 params.cols * params.cell_size)
		self.setWindowTitle("SnakeBot")

		self.game_widget = GameWidget(env, params, self)
		self.setCentralWidget(self.game_widget)

		self.create_widget = CreateWindow(self)
		self.agent_widget = AgentWidget(agent, self)

		self.__create_menu()

		self.env = env
		self.params = params
		self.is_running = True

		self.timer = QBasicTimer()
		self.timer.start(params.update_rate, self)

		self.current_state = None
		self.previous_state = None
		self.previous_action = None

		self.center()
		self.update()

	def keyPressEvent(self, event):
		if event.key() == Qt.Key_Space:
			self.previous_state = self.current_state
			self.current_state = self.env.start_new_episode()
			self.is_running = True
			self.restart_timer()
		elif event.key() == Qt.Key_P:
			self.is_running = not self.is_running
			if self.is_running:
				self.restart_timer()
		else:
			self.agent_widget.keyPressEvent(event)

	def timerEvent(self, event):
		if not self.is_running:
			self.timer.stop()

		if event.timerId() == self.timer.timerId():
			# FIXME: This repaint fixes issues that happens upon death but makes each move delayed by one frame.
			self.repaint()
			self.previous_state = self.current_state
			self.previous_action = self.agent_widget.get_action(self.current_state)
			self.current_state, _ = self.env.step(self.previous_action)
			self.is_running = not self.env.episode_is_done()
			# self.repaint()

	def center(self):
		frame_gm = self.frameGeometry()
		screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
		center_point = QApplication.desktop().screenGeometry(screen).center()
		frame_gm.moveCenter(center_point)
		self.move(frame_gm.topLeft())

	def restart_timer(self):
		self.timer.start(self.params.update_rate, self)

	def __create_menu(self):
		create_action = QAction("&Create", self)
		create_action.setShortcut("Ctrl+C")
		create_action.setStatusTip('Create Agent')
		create_action.triggered.connect(self.__on_push_create)

		import_action = QAction("&Import", self)
		import_action.setShortcut("Ctrl+I")
		import_action.setStatusTip('Import Agent')
		import_action.triggered.connect(self.__on_push_import)

		export_action = QAction("&Export", self)
		export_action.setShortcut("Ctrl+L")
		export_action.setStatusTip('Export Agent')
		export_action.triggered.connect(self.__on_push_export)

		quit_action = QAction("&Quit", self)
		quit_action.setShortcut("Ctrl+Q")
		quit_action.setStatusTip('Quit Application')
		quit_action.triggered.connect(self.__on_push_quit)

		run_action = QAction("&Run", self)
		run_action.setShortcut("Ctrl+R")
		run_action.setStatusTip('Run Agent')
		run_action.triggered.connect(self.__on_push_run)

		train_action = QAction("&Train", self)
		train_action.setShortcut("Ctrl+T")
		train_action.setStatusTip('Train Agent')
		train_action.triggered.connect(self.__on_push_train)

		reward_stat_action = QAction("&Reward", self)
		reward_stat_action.setStatusTip('Rewards Over Time')
		reward_stat_action.triggered.connect(self.__on_push_reward_stat)

		action_stat_action = QAction("&Action", self)
		action_stat_action.setStatusTip('Actions Over Episodes')
		action_stat_action.triggered.connect(self.__on_push_action_stat)

		exploration_exploitation_stat_exploration_exploitation = QAction("&Exploration/Exploitation", self)
		exploration_exploitation_stat_exploration_exploitation.setStatusTip(
			'Exploration/Exploitation Percentages Over Episodes')
		exploration_exploitation_stat_exploration_exploitation.triggered.connect(
			self.__on_push_exploration_exploitation_stat)

		main_menu = self.menuBar()
		file_menu = main_menu.addMenu("&File")
		file_menu.addAction(create_action)
		file_menu.addAction(import_action)
		file_menu.addAction(export_action)
		file_menu.addAction(quit_action)

		file_menu = main_menu.addMenu("&Agent")
		file_menu.addAction(run_action)
		file_menu.addAction(train_action)

		file_menu = main_menu.addMenu("&Stats")
		file_menu.addAction(reward_stat_action)
		file_menu.addAction(action_stat_action)
		file_menu.addAction(exploration_exploitation_stat_exploration_exploitation)

	@pyqtSlot()
	def __on_push_create(self):
		# self.create_widget.show()
		print("Create Agent")

	@pyqtSlot()
	def __on_push_import(self):
		filename = QFileDialog.getOpenFileName(self, "Import Model",
											   os.getcwd(), "Pickle files (*.p)")[0]
		filename_parts = filename.split("/")[-1].split("_")

		model = read_model(filename)

		state_class_name = filename_parts[0]
		reward_class_name = filename_parts[1]

		state_module = importlib.import_module("src.snake.state")
		reward_module = importlib.import_module("src.snake.reward")

		state_class = getattr(state_module, state_class_name)
		reward_class = getattr(reward_module, reward_class_name)

		self.params.state = state_class
		self.params.reward = reward_class

		self.game_widget = GameWidget(self.env, self.params, self)
		self.setCentralWidget(self.game_widget)

		agent = SnakeAgent(policy=EpsilonGreedyPolicy(self.env, 0),
						   action_value_function=model.Q)

		self.agent_widget = AgentWidget(agent=agent, parent=self)

	@pyqtSlot()
	def __on_push_export(self):
		print("Export Agent")

	@pyqtSlot()
	def __on_push_quit(self):
		sys.exit()

	@pyqtSlot()
	def __on_push_run(self):
		print("Run Agent")

	@pyqtSlot()
	def __on_push_train(self):
		print("Train Agent")

	@pyqtSlot()
	def __on_push_reward_stat(self):
		print("Reward Stat")

	@pyqtSlot()
	def __on_push_action_stat(self):
		print("Action Stat")

	@pyqtSlot()
	def __on_push_exploration_exploitation_stat(self):
		print("Exploration/Exploitation Stat")


if __name__ == '__main__':
	params = SnakeParameters()
	env = SnakeEnvironment(params)
	player = SnakePlayer()

	app = QApplication(["SnakeBot"])
	window = Window(env, player, params)
	window.show()
	app.exec_()
