import pickle

import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as timer

from src.algorithms.sarsa import Sarsa
from src.core.policy import EpsilonGreedyPolicy
from src.experiment.plot import plot_rewards_over_time, plot_exploration_vs_exploitation_over_time, \
	plot_actions_over_time, plot_average_reward_over_time, plot_average_actions_over_time
from src.main import start_app
from src.snake.agent import SnakeAgent
from src.snake.environment import SnakeEnvironment
from src.snake.parameters import SnakeParameters


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

	pickle.dump(learner.Q, open("../../cache/sarsa_%s.p" % params.file_str, "wb"))

	print("Elapsed time:", timer() - start)


def run_sarsa():
	params = SnakeParameters()
	env = SnakeEnvironment(params)
	value_function = pickle.load(open("../../cache/sarsa_%s.p" % params.file_str, "rb"))

	agent = SnakeAgent(policy=EpsilonGreedyPolicy(env, 0),
					   action_value_function=value_function)

	start_app(env, agent, params)


def main():
	np.random.seed(123)

	start = timer()

	params = SnakeParameters()
	env = SnakeEnvironment(params)
	policy = EpsilonGreedyPolicy(env, params.epsilon)

	learner = Sarsa(action_value_function=params.value_function,
					policy=policy,
					learning_rate=params.learning_rate,
					discount_factor=params.discount_factor)

	learner.train(env, params.train_episodes)

	plt.figure(1)

	plt.subplot(221)
	plot_rewards_over_time(learner)

	plt.subplot(222)
	plot_average_reward_over_time(learner)

	plt.subplot(223)
	plot_actions_over_time(learner)

	plt.subplot(224)
	plot_average_actions_over_time(learner)

	plt.show()

	print("Elapsed time:", timer() - start)


if __name__ == "__main__":
	main()
