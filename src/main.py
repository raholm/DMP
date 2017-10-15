from PyQt5.QtWidgets import QApplication

from src.algorithms.qlearning import QLearning
from src.core.discount_factor import StaticDiscountFactor
from src.core.policy import EpsilonGreedyPolicy, GreedyPolicy
from src.core.value_function import DictActionValueFunction
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


def test_train_qlearning():
	params = SnakeParameters()
	env = SnakeEnvironment(params)

	discount_factor = StaticDiscountFactor(1)
	learning_rate = 0.5
	policy = EpsilonGreedyPolicy(env, 0.25)
	value_function = DictActionValueFunction(0)

	learning_alg = QLearning(action_value_function=value_function,
							 policy=policy,
							 learning_rate=learning_rate,
							 discount_factor=discount_factor)

	learning_alg.train(env, n_episodes=100000)

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
	test_train_qlearning()
