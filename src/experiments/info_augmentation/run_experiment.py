import os

import numpy as np
from matplotlib import pyplot as plt

from src.algorithms.qlearning import QLearning
from src.experiments.experiment import setup_experiment, test_experiment, get_models_from_experiment, \
	run_experiment_in_simulator
from src.experiments.global_settings import get_seeds
from src.experiments.plot import plot_multi_average_game_score_over_time
from src.experiments.reward.run_experiment import get_test_seed
from src.snake.agent import SnakeRandomAgent
from src.snake.reward import ZeroTravelPosScore, NegTravelPosScore
from src.snake.state import BoardState, BoardScoreState, BoardDimensionState, BoardDimensionScoreState, \
	DirectionalScoreState
from src.util.io import get_project_path
from src.util.util import get_state_class_from_string, get_reward_class_from_string, create_dir


def get_experiment():
	experiment = setup_experiment(QLearning, "info_augmentation", get_seeds()[0])
	experiment.train_episodes = 1000000
	return experiment


def get_states():
	return [BoardState,
			BoardScoreState,
			BoardDimensionState,
			BoardDimensionScoreState]


def get_rewards():
	return [ZeroTravelPosScore]


def experiment_board_state_with_and_without_extra_information():
	def _get_models(experiment, states, rewards):
		models, states, rewards, _ = get_models_from_experiment(experiment,
																states,
																rewards,
																train_if_missing=True)
		return models, states, rewards

	def _plot_comparison(experiment, models, states):
		game_score_per_episode = []

		for model in models:
			game_score_per_episode.append(experiment.env.game_score_coef * model.food_count_per_episode[::10])

		x = np.arange(1, (len(game_score_per_episode[0]) * 10) + 1, 10)

		output_dir = os.path.join(get_project_path(),
								  "images",
								  "qlearning",
								  experiment.experiment_type,
								  str(experiment.seed))
		create_dir(output_dir)
		filename = "board_state_average_game_score_over_time.png"
		output = os.path.join(output_dir, filename)

		plot_multi_average_game_score_over_time(x, game_score_per_episode, states)
		plt.savefig(output)

	# plt.show()

	def _compare_test_performance(experiment, states, rewards):
		states = list(map(get_state_class_from_string, np.unique(states)))
		rewards = list(map(get_reward_class_from_string, np.unique(rewards)))
		n_episodes = 1000

		test_results = test_experiment(experiment, n_episodes=n_episodes,
									   states=states, rewards=rewards)

		print("Test Performance on %i episodes: " % (n_episodes,))
		for (state, reward), scores in test_results.items():
			print("%s: Mean Score %0.03f, Std %0.03f" %
				  (state.__name__,
				   np.mean(scores),
				   np.std(scores)), )

		agent = SnakeRandomAgent(experiment.env)
		scores = experiment.env.run(agent, n_episodes)
		print("%s: Mean Score %0.03f, Std %0.03f" %
			  ("Random Agent",
			   np.mean(scores),
			   np.std(scores)))

	experiment = get_experiment()
	states = get_states()
	rewards = get_rewards()

	run_experiment_in_simulator(experiment, states[1], rewards[0])


# models, states, rewards = _get_models(experiment, states, rewards)
# _plot_comparison(experiment, models, states)
# _compare_test_performance(experiment, states, rewards)


def main():
	# experiment_board_state_with_and_without_extra_information()
	experiment = setup_experiment(QLearning, "info_augmentation", get_test_seed())
	experiment.train_episodes = 17000
	experiment.state = DirectionalScoreState
	experiment.reward = NegTravelPosScore

	experiment.load()
	experiment.train()
	experiment.save()

	run_experiment_in_simulator(experiment)


if __name__ == '__main__':
	main()
