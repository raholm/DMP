import pickle

from PyQt5.QtWidgets import QApplication
from timeit import default_timer as timer

from src.algorithms.qlearning import QLearning
from src.algorithms.sarsa import Sarsa
from src.core.policy import EpsilonGreedyPolicy
from src.snake.gui import Window
from src.snake.parameters import SnakeParameters
from src.snake.agent import SnakePlayer, SnakeAgent
from src.snake.environment import SnakeEnvironment
from src.snake.state import WholeState, DistanceState, SnakeAndFoodState


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

	pickle.dump(learner.Q, open("../cache/qlearning_%s.p" % params.file_str, "wb"))

	print("Elapsed time:", timer() - start)


def run_qlearning():
	params = SnakeParameters()
	env = SnakeEnvironment(params)
	value_function = pickle.load(open("../cache/qlearning_%s.p" % params.file_str, "rb"))

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

	pickle.dump(learner.Q, open("../cache/sarsa_%s.p" % params.file_str, "wb"))

	print("Elapsed time:", timer() - start)


def run_sarsa():
	params = SnakeParameters()
	env = SnakeEnvironment(params)
	value_function = pickle.load(open("../cache/sarsa_%s.p" % params.file_str, "rb"))

	agent = SnakeAgent(policy=EpsilonGreedyPolicy(env, 0),
					   action_value_function=value_function)

	start_app(env, agent, params)


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

	start_app(env, player, params)


if __name__ == "__main__":
	# main()
	# train_qlearning()
	# run_qlearning()

	train_sarsa()
	run_sarsa()


