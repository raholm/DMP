import pickle
import os

import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as timer

from src.algorithms.sarsa import Sarsa
from src.core.policy import EpsilonGreedyPolicy
from src.experiment.plot import plot_rewards_over_time, plot_actions_over_time, plot_average_reward_over_time, \
	plot_average_actions_over_time, plot_multi_average_reward_over_time, plot_multi_average_actions_over_time
from src.main import start_app
from src.snake.agent import SnakeAgent
from src.snake.environment import SnakeEnvironment
from src.snake.parameters import SnakeParameters
from src.snake.state import SnakeAndFoodWithScoreState, BoardState, SnakeAndFoodWithoutScoreState, DistanceState, \
	DirectionalDistanceState
from src.util.io import write_learner, read_learner


def train_and_store_sarsa_model(env, params, dir):
	learner = Sarsa(action_value_function=params.value_function,
					policy=params.policy,
					learning_rate=params.learning_rate,
					discount_factor=params.discount_factor)

	start = timer()

	learner.train(env, params.train_episodes)

	print("Training time:", timer() - start)

	write_learner(learner, os.path.join(dir, "%s.p" % params.file_str))

	return learner


def train_sarsa_models():
	np.random.seed(123)

	start = timer()

	params = SnakeParameters()
	env = SnakeEnvironment(params)
	params.policy = EpsilonGreedyPolicy(env, params.epsilon)
	dir = "../../models/sarsa/state"

	# Model 1
	params.state = BoardState
	train_and_store_sarsa_model(env, params, dir)

	# Model 2
	params.state = SnakeAndFoodWithScoreState
	train_and_store_sarsa_model(env, params, dir)

	# Model 3
	params.state = SnakeAndFoodWithoutScoreState
	train_and_store_sarsa_model(env, params, dir)

	# Model 4
	params.state = DistanceState
	train_and_store_sarsa_model(env, params, dir)

	# Model 5
	params.state = DirectionalDistanceState
	train_and_store_sarsa_model(env, params, dir)

	print("Elapsed time:", timer() - start)


def read_sarsa_models(dir):
	filenames = []
	models = []

	for subdir, dirs, files in os.walk(dir):
		for file in files:
			file_path = os.path.join(subdir, file)
			filenames.append(file)
			models.append(read_learner(file_path))

	return models, filenames


def analyze_sarsa_models():
	models, filenames = read_sarsa_models("../../models/sarsa/state")

	actions_per_episode = []
	rewards_per_episode = []
	states = []

	for filename in filenames:
		states.append(filename.split("_")[0])

	for model in models:
		actions_per_episode.append(model.actions_per_episode)
		rewards_per_episode.append(model.rewards_per_episode)

	plt.figure(1)

	plt.subplot(121)
	plot_multi_average_reward_over_time(rewards_per_episode, states)

	plt.subplot(122)
	plot_multi_average_actions_over_time(actions_per_episode, states)

	plt.show()


def run_sarsa():
	params = SnakeParameters()
	env = SnakeEnvironment(params)
	value_function = pickle.load(open("../../models/sarsa_%s.p" % params.file_str, "rb"))

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
	# train_sarsa_models()
	analyze_sarsa_models()
