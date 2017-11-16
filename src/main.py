import pickle

import os

import sys
from PyQt5 import QtGui
from timeit import default_timer as timer

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, QFileDialog

from src.algorithms.qlearning import QLearning
from src.algorithms.sarsa import Sarsa
from src.core.policy import EpsilonGreedyPolicy
from src.snake.agent import SnakePlayer, SnakeAgent
from src.snake.environment import SnakeEnvironment
from src.snake.gui import Window
from src.snake.parameters import SnakeParameters
from src.snake.state import BoardState, SnakeFoodState, DirectionalState, DirectionalDistanceState


def start_app(env, agent, params):
	app = QApplication(["SnakeBot"])
	window = Window(env, agent, params)
	window.show()
	app.exec_()


def train_qlearning():
	start = timer()

	params = SnakeParameters()
	env = SnakeEnvironment(params)
	policy = EpsilonGreedyPolicy(env, params.epsilon)

	learner = QLearning(action_value_function=params.value_function,
						policy=policy,
						learning_rate=params.learning_rate,
						discount_factor=params.discount_factor)

	learner.train(env, params.train_episodes)

	pickle.dump(learner.Q, open("../models/qlearning_%s.p" % params.file_str, "wb"))

	print("Elapsed time:", timer() - start)


def run_qlearning():
	params = SnakeParameters()
	env = SnakeEnvironment(params)
	value_function = pickle.load(open("../models/qlearning_%s.p" % params.file_str, "rb"))

	agent = SnakeAgent(policy=EpsilonGreedyPolicy(env, 0),
					   action_value_function=value_function)

	start_app(env, agent, params)


def train_sarsa():
	start = timer()

	params = SnakeParameters()
	env = SnakeEnvironment(params)
	policy = EpsilonGreedyPolicy(env, params.epsilon)

	learner = Sarsa(action_value_function=params.value_function,
					policy=policy,
					learning_rate=params.learning_rate,
					discount_factor=params.discount_factor)

	learner.train(env, params.train_episodes)

	pickle.dump(learner.Q, open("../models/sarsa_%s.p" % params.file_str, "wb"))

	print("Elapsed time:", timer() - start)


def run_sarsa():
	params = SnakeParameters()
	env = SnakeEnvironment(params)
	value_function = pickle.load(open("../models/sarsa_%s.p" % params.file_str, "rb"))

	agent = SnakeAgent(policy=EpsilonGreedyPolicy(env, 0),
					   action_value_function=value_function)

	start_app(env, agent, params)


def test_state():
	params = SnakeParameters()
	env = SnakeEnvironment(params)

	state = BoardState(env)
	print(state)
	print(hash(state))

	state = DirectionalState(env)
	print(state)
	print(hash(state))

	state = SnakeFoodState(env)
	print(state)
	print(hash(state))

	state = DirectionalDistanceState(env)
	print(state)
	print(hash(state))


def main():
	params = SnakeParameters()
	env = SnakeEnvironment(params)
	player = SnakePlayer()

	start_app(env, player, params)


if __name__ == "__main__":
	main()
# test_state()
# train_qlearning()
# run_qlearning()

# train_sarsa()
# run_sarsa()
