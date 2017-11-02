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
from src.snake.state import BoardState, \
	DirectionalDistanceState, BoardDimensionScoreState, BoardDimensionState, BoardScoreState, SnakeFoodState, \
	SnakeFoodDimensionState, SnakeFoodScoreState, SnakeFoodDimensionScoreState, DirectionalState, \
	DirectionalDimensionState, DirectionalScoreState, DirectionalDimensionScoreState, DirectionalDistanceDimensionState, \
	DirectionalDistanceScoreState, DirectionalDistanceDimensionScoreState
from src.util.io import write_learner, read_learner


def train_and_store_model(env, params, dir):
	learner = Sarsa(action_value_function=params.value_function,
					policy=params.policy,
					learning_rate=params.learning_rate,
					discount_factor=params.discount_factor)

	start = timer()

	learner.train(env, params.train_episodes)

	print("Training time:", timer() - start)

	write_learner(learner, os.path.join(dir, "%s.p" % params.file_str))

	return learner


def train_state_models(states, env, params, dir):
	for state in states:
		params.state = state
		train_and_store_model(env, params, dir)


def train_board_state_models(env, params, dir):
	states = [BoardState, BoardDimensionState,
			  BoardScoreState, BoardDimensionScoreState]
	train_state_models(states, env, params, dir)


def train_snake_food_state_models(env, params, dir):
	states = [SnakeFoodState, SnakeFoodDimensionState,
			  SnakeFoodScoreState, SnakeFoodDimensionScoreState]
	train_state_models(states, env, params, dir)


def train_directional_state_models(env, params, dir):
	states = [DirectionalState, DirectionalDimensionState,
			  DirectionalScoreState, DirectionalDimensionScoreState]
	train_state_models(states, env, params, dir)


def train_directional_distance_state_models(env, params, dir):
	states = [DirectionalDistanceState, DirectionalDistanceDimensionState,
			  DirectionalDistanceScoreState, DirectionalDistanceDimensionScoreState]
	train_state_models(states, env, params, dir)


def train_models():
	np.random.seed(123)

	start = timer()

	params = SnakeParameters()
	env = SnakeEnvironment(params)
	params.policy = EpsilonGreedyPolicy(env, params.epsilon)
	dir = "../../models/sarsa/state"

	train_board_state_models(env, params, dir)
	train_snake_food_state_models(env, params, dir)
	train_directional_state_models(env, params, dir)
	train_directional_distance_state_models(env, params, dir)

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


def analyze_state_models(models, states):
	actions_per_episode = []
	rewards_per_episode = []

	for model in models:
		actions_per_episode.append(model.actions_per_episode)
		rewards_per_episode.append(model.rewards_per_episode)

	plt.figure(1)

	plt.subplot(121)
	plot_multi_average_reward_over_time(rewards_per_episode, states)

	plt.subplot(122)
	plot_multi_average_actions_over_time(actions_per_episode, states)

	plt.show()


def filter_state_models(models, states, predicate):
	current_models = []
	current_states = []

	for i, state in enumerate(states):
		if predicate(state):
			current_models.append(models[i])
			current_states.append(state)

	return current_models, current_states


def analyze_board_state_models(models, states):
	current_models, current_states = \
		filter_state_models(models, states,
							lambda state: state.startswith("Board"))
	analyze_state_models(current_models, current_states)


def analyze_snake_food_state_models(models, states):
	current_models, current_states = \
		filter_state_models(models, states,
							lambda state: state.startswith("SnakeFood"))
	analyze_state_models(current_models, current_states)


def analyze_directional_state_models(models, states):
	current_models, current_states = \
		filter_state_models(models, states,
							lambda state: state.startswith("Directional") and
										  not state.startswith("DirectionalDistance"))
	analyze_state_models(current_models, current_states)


def analyze_directional_distance_state_models(models, states):
	current_models, current_states = \
		filter_state_models(models, states,
							lambda state: state.startswith("DirectionalDistance"))
	analyze_state_models(current_models, current_states)


def analyze_models():
	models, filenames = read_sarsa_models("../../models/sarsa/state")

	states = []

	for filename in filenames:
		states.append(filename.split("_")[0])

	# analyze_board_state_models(models, states)
	# analyze_snake_food_state_models(models, states)
	# analyze_directional_state_models(models, states)
	analyze_directional_distance_state_models(models, states)


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
	# train_models()
	analyze_models()
