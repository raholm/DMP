import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.qlearning import QLearning
from src.experiments.experiment import setup_experiment, train_experiment, get_models_from_experiment, \
	run_experiment_in_simulator
from src.experiments.global_settings import get_seeds
from src.experiments.plot import plot_multi_average_reward_over_time, plot_multi_average_food_count_over_time, \
	plot_multi_average_game_score_over_time
from src.snake.reward import NegTravelPosScore, PosTravelPosScore, NegDistancePosBodySize, \
	NegDistanceNegSelfCollisionPosBodySize, NegDistanceNegBorderCollisionPosBodySize, \
	ZeroTravelPosScore, NegTravelNegBorderCollisionPosScore
from src.snake.state import BoardScoreState, DirectionalScoreState
from src.util.util import filter_models_with_rewards_by_state


def get_test_seed():
	return 1337


def get_states():
	return [BoardScoreState,
			DirectionalScoreState]


def get_rewards():
	return [NegTravelPosScore,
			ZeroTravelPosScore,
			PosTravelPosScore,
			NegDistancePosBodySize,
			NegDistanceNegSelfCollisionPosBodySize,
			NegDistanceNegBorderCollisionPosBodySize,
			NegTravelNegBorderCollisionPosScore]


def get_experiment():
	experiment = setup_experiment(QLearning, "reward", get_seeds()[0])
	experiment.train_episodes = 1000000
	return experiment


def get_testing_experiment():
	experiment = setup_experiment(QLearning, "reward", get_test_seed[0])
	experiment.train_episodes = 1000
	return experiment


def train_models():
	train_experiment(get_experiment(), get_states(), get_rewards())


def compare_models():
	experiment = get_experiment()
	models, states, rewards, _ = get_models_from_experiment(experiment,
															get_states(),
															get_rewards())

	models, states, rewards = \
		filter_models_with_rewards_by_state(models, states, rewards,
											lambda state:
											state.startswith("Board"))

	rewards_per_episode = []
	game_score_per_episode = []

	for model in models:
		rewards_per_episode.append(model.rewards_per_episode[::10])
		game_score_per_episode.append(experiment.env.game_score_coef * model.food_count_per_episode[::10])

	x = np.arange(1, (len(rewards_per_episode[0]) * 10) + 1, 10)

	plt.subplot(211)
	plot_multi_average_reward_over_time(x, rewards_per_episode, rewards)

	plt.subplot(212)
	plot_multi_average_game_score_over_time(x, game_score_per_episode, rewards)

	plt.show()


def main():
	# train_models()
	# compare_models()
	experiment = setup_experiment(QLearning, "reward", get_seeds()[0])
	experiment.train_episodes = 1000000
	experiment.state = get_states()[1]
	experiment.reward = get_rewards()[-2]

	experiment.load()
	experiment.train()
	experiment.save()

	run_experiment_in_simulator(experiment)

	return


# experiment = setup_experiment(QLearning, "reward", get_seeds()[0])
# experiment.train_episodes = 10000
#
# state = DirectionalDimensionScoreState
# reward = NegTravelPosScore
#
# experiment.state = state
# experiment.reward = reward
# experiment.train_episodes = 10000
#
# if experiment.has_cached_model():
# 	print("Model is cached!")
# else:
# 	print("Model is not cached!")
# 	train_experiment(experiment)
#
# experiment.load()
#
# result = test_experiment(experiment, 10000)
#
# for (state, reward), value in result.items():
# 	print(state.__name__)
# 	print(reward.__name__)
# 	print(np.mean(value["food"]))
#
# run_experiment_in_simulator(experiment)


if __name__ == '__main__':
	main()
