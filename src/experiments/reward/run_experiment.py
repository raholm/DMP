import os

import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.qlearning import QLearning
from src.algorithms.sarsa import Sarsa
from src.core.policy import GreedyPolicy
from src.experiments.experiment import setup_experiment, train_experiment
from src.experiments.global_settings import get_seeds
from src.experiments.plot import plot_multi_average_game_score_over_time
from src.experiments.state.run_experiment import aggregate_models_by_avg
from src.snake.agent import SnakeAgent
from src.snake.reward import NegTravelPosScore, PosTravelPosScore, NegDistanceNegBorderCollisionPosBodySize, \
	ZeroTravelPosScore, NegTravelNegBorderCollisionPosScore
from src.snake.state import BoardScoreState, DirectionalScoreState
from src.util.io import get_project_path
from src.util.math import compute_mean_over_time, compute_correlation
from src.util.util import filter_models_with_rewards_by_state, create_dir, get_state_class_from_string, \
	get_reward_class_from_string


def get_test_seed():
	return 1337


def get_states():
	return [BoardScoreState,
			DirectionalScoreState]


def get_rewards():
	return [NegTravelPosScore,
			ZeroTravelPosScore,
			PosTravelPosScore,
			NegDistanceNegBorderCollisionPosBodySize,
			NegTravelNegBorderCollisionPosScore]


def get_qlearning_experiment(params):
	return setup_experiment(QLearning, "reward", params["seed"])


def get_sarsa_experiment(params):
	return setup_experiment(Sarsa, "reward", params["seed"])


def get_testing_experiment():
	experiment = setup_experiment(QLearning, "reward", get_test_seed[0])
	experiment.train_episodes = 1000
	return experiment


def train_qlearning_models():
	params = {"seed": None}

	for seed in get_seeds():
		params["seed"] = seed
		train_experiment(get_qlearning_experiment(params), get_states(), get_rewards())


def train_sarsa_models():
	params = {"seed": None}

	for seed in get_seeds():
		params["seed"] = seed
		train_experiment(get_sarsa_experiment(params), get_states(), get_rewards())


def experiment01(experiment, models, states, rewards, params):
	rewards_per_episode = []
	game_score_per_episode = []

	for model in models:
		rewards_per_episode.append(model.rewards_per_episode[::10])
		game_score_per_episode.append(experiment.env.game_score_coef * model.food_count_per_episode[::10])

	x = np.arange(1, (len(rewards_per_episode[0]) * 10) + 1, 10)

	plot_multi_average_game_score_over_time(x, game_score_per_episode, rewards)
	create_dir("/".join(params["image_output"].split("/")[:-1]))
	plt.savefig(params["image_output"])
	# plt.show()
	plt.close()

	average_reward_over_time = map(compute_mean_over_time, rewards_per_episode)
	average_game_score_over_time = map(compute_mean_over_time, game_score_per_episode)
	corr_coefs = compute_correlation(average_game_score_over_time,
									 average_reward_over_time)

	print("Average Correlation between Reward and Game Score: %0.03f, std: %0.03f" %
		  (np.mean(corr_coefs), np.std(corr_coefs)))
	print("Average Abs. Correlation between Reward and Game Score: %0.03f, std: %0.03f" %
		  (np.mean(np.abs(corr_coefs)), np.std(np.abs(corr_coefs))))

	for model, state, reward in zip(models, states, rewards):
		state_class = get_state_class_from_string(state)
		reward_class = get_reward_class_from_string(rewards[0])
		experiment.state = state_class
		experiment.reward = reward_class

		print("Test Performance of %s %s" % (state, reward,))
		agent = SnakeAgent(GreedyPolicy(experiment.env),
						   model.Q)
		scores = experiment.env.run(agent, 10000)
		print("Avg Score: %0.03f (Std: %0.03f)" % (np.mean(scores), np.std(scores)))


def experiment01_qlearning_board_state():
	def get_params(seed):
		image_output = os.path.join(get_project_path(),
									"images",
									"qlearning",
									"reward",
									str(seed),
									"reward_qlearning_board_state_average_game_score_over_time.png")

		return {"seed": seed,
				"image_output": image_output}

	params = get_params(42)
	experiment = get_qlearning_experiment(params)
	models, states, rewards = aggregate_models_by_avg(get_qlearning_experiment, get_seeds(), get_params, get_states,
													  get_rewards)

	models, states, rewards = \
		filter_models_with_rewards_by_state(models, states, rewards,
											lambda state:
											state.startswith("Board"))
	experiment01(experiment, models, states, rewards, params)


def experiment01_qlearning_directional_state():
	def get_params(seed):
		image_output = os.path.join(get_project_path(),
									"images",
									"qlearning",
									"reward",
									str(seed),
									"reward_qlearning_directional_state_average_game_score_over_time.png")

		return {"seed": seed,
				"image_output": image_output}

	params = get_params(42)
	experiment = get_qlearning_experiment(params)
	models, states, rewards = aggregate_models_by_avg(get_qlearning_experiment, get_seeds(), get_params, get_states,
													  get_rewards)

	models, states, rewards = \
		filter_models_with_rewards_by_state(models, states, rewards,
											lambda state:
											state.startswith("Directional"))
	experiment01(experiment, models, states, rewards, params)


def experiment01_sarsa_board_state():
	def get_params(seed):
		image_output = os.path.join(get_project_path(),
									"images",
									"sarsa",
									"reward",
									str(seed),
									"reward_sarsa_board_state_average_game_score_over_time.png")

		return {"seed": seed,
				"image_output": image_output}

	params = get_params(42)
	experiment = get_sarsa_experiment(params)
	models, states, rewards = aggregate_models_by_avg(get_sarsa_experiment, get_seeds(), get_params, get_states,
													  get_rewards)

	models, states, rewards = \
		filter_models_with_rewards_by_state(models, states, rewards,
											lambda state:
											state.startswith("Board"))
	experiment01(experiment, models, states, rewards, params)


def experiment01_sarsa_directional_state():
	def get_params(seed):
		image_output = os.path.join(get_project_path(),
									"images",
									"sarsa",
									"reward",
									str(seed),
									"reward_sarsa_directional_state_average_game_score_over_time.png")

		return {"seed": seed,
				"image_output": image_output}

	params = get_params(42)
	experiment = get_qlearning_experiment(params)
	models, states, rewards = aggregate_models_by_avg(get_sarsa_experiment, get_seeds(), get_params, get_states,
													  get_rewards)

	models, states, rewards = \
		filter_models_with_rewards_by_state(models, states, rewards,
											lambda state:
											state.startswith("Directional"))
	experiment01(experiment, models, states, rewards, params)


def main():
	experiment01_qlearning_board_state()
	experiment01_qlearning_directional_state()
	experiment01_sarsa_board_state()
	experiment01_sarsa_directional_state()


if __name__ == '__main__':
	main()
