import os

import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.qlearning import QLearning
from src.algorithms.sarsa import Sarsa
from src.core.policy import GreedyPolicy
from src.experiments.experiment import setup_experiment, get_models_from_experiment, train_experiment
from src.experiments.global_settings import get_seeds
from src.experiments.plot import plot_multi_average_game_score_over_time
from src.snake.agent import SnakeAgent
from src.snake.reward import ZeroTravelPosScore
from src.snake.state import BoardScoreState, SnakeFoodScoreState, DirectionalScoreState, ShortestPathScoreState
from src.util.io import get_project_path
from src.util.math import compute_mean_over_time, compute_correlation
from src.util.util import create_dir


def get_rewards():
	return [ZeroTravelPosScore]


def get_states():
	return [BoardScoreState,
			SnakeFoodScoreState,
			DirectionalScoreState,
			ShortestPathScoreState]


def get_qlearning_experiment(params):
	return setup_experiment(QLearning, "state", params["seed"])


def get_sarsa_experiment(params):
	return setup_experiment(Sarsa, "state", params["seed"])


def train_qlearning_models(params):
	train_experiment(get_qlearning_experiment(params), get_states(), get_rewards())


def train_sarsa_models(params):
	train_experiment(get_sarsa_experiment(params), get_states(), get_rewards())


def aggregate_models_by_avg(seeds, params_getter):
	aggregated_models = {}

	for seed in seeds:
		experiment = get_sarsa_experiment(params_getter(seed))

		models, states, rewards, _ = get_models_from_experiment(experiment,
																get_states(),
																get_rewards(),
																train_if_missing=True)

		for model, state, reward in zip(models, states, rewards):
			key = (state, reward)

			if key not in aggregated_models:
				aggregated_models[key] = model
			else:
				aggregated_models[key].food_count_per_episode += model.food_count_per_episode
				aggregated_models[key].rewards_per_episode += model.rewards_per_episode

		for seed in get_seeds()[1:]:
			params = params_getter(seed)
			tmp_experiment = get_sarsa_experiment(params)

			tmp_models, tmp_states, tmp_rewards, _ = get_models_from_experiment(
				tmp_experiment,
				get_states(),
				get_rewards(),
				train_if_missing=True)

			for true_model, true_state in zip(models, states):
				for tmp_model, tmp_state in zip(models, states):
					if true_state == tmp_state:
						true_model.food_count_per_episode += tmp_model.food_count_per_episode
						true_model.rewards_per_episode += tmp_model.rewards_count_per_episode

		for _, model in aggregated_models.items():
			model.food_count_per_episode = model.food_count_per_episode / len(seeds)
			model.rewards_per_episode = model.rewards_per_episode / len(seeds)

	models, states, rewards = [], [], []

	for (state, reward), model in aggregated_models.keys():
		models.append(model)
		states.append(state)
		rewards.append(reward)

	return models, states, rewards


def experiment01(experiment, models, states, rewards, params):
	rewards_per_episode = []
	game_score_per_episode = []

	for model in models:
		rewards_per_episode.append(model.rewards_per_episode[::10])
		game_score_per_episode.append(experiment.env.game_score_coef * model.food_count_per_episode[::10])

	x = np.arange(1, (len(rewards_per_episode[0]) * 10) + 1, 10)

	plot_multi_average_game_score_over_time(x, game_score_per_episode, states)

	create_dir("/".join(params["image_output"].split("/")[:-1]))
	plt.savefig(params["image_output"])
	# plt.show()

	average_reward_over_time = map(compute_mean_over_time, rewards_per_episode)
	average_game_score_over_time = map(compute_mean_over_time, game_score_per_episode)

	corr_coefs = compute_correlation(average_game_score_over_time,
									 average_reward_over_time)

	print("Average Correlation between Reward and Game Score: %0.03f, std: %0.03f" %
		  (np.mean(corr_coefs), np.std(corr_coefs)))
	print("Average Abs. Correlation between Reward and Game Score: %0.03f, std: %0.03f" %
		  (np.mean(np.abs(corr_coefs)), np.std(np.abs(corr_coefs))))

	for model, state in zip(models, states):
		print("Test Performance of %s" % (state,))
		agent = SnakeAgent(GreedyPolicy(experiment.env),
						   experiment.model_.Q)
		scores = experiment.env.run(agent, 10000)
		print("Avg Score: %0.03f (Std: %0.03f)" % (np.mean(scores), np.std(scores)))


def experiment01_qlearning():
	def get_params(seed):
		image_output = os.path.join(get_project_path(),
									"images",
									"qlearning",
									"state",
									str(seed),
									"state_qlearning_average_game_score_over_time.png")

		return {"seed": seed,
				"image_output": image_output}

	params = get_params(42)
	experiment = get_sarsa_experiment(params)
	models, states, rewards = aggregate_models_by_avg(get_seeds(), get_params)

	experiment01(experiment, models, states, rewards, params)


def experiment01_sarsa():
	def get_params(seed):
		image_output = os.path.join(get_project_path(),
									"images",
									"sarsa",
									"state",
									str(seed),
									"state_sarsa_average_game_score_over_time.png")

		return {"seed": seed,
				"image_output": image_output}

	params = get_params(42)
	experiment = get_sarsa_experiment(params)
	models, states, rewards = aggregate_models_by_avg(get_seeds(), get_params)

	experiment01(experiment, models, states, rewards, params)


def main():
	experiment01_qlearning()
	experiment01_sarsa()


if __name__ == '__main__':
	main()
