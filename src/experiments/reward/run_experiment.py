import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.qlearning import QLearning
from src.core.policy import GreedyPolicy
from src.experiments.experiment import setup_experiment
from src.experiments.plot import plot_multi_average_reward_over_time, plot_multi_average_food_count_over_time
from src.main import start_app
from src.snake.agent import SnakeAgent
from src.snake.reward import NegTravelPosScore, PosTravelPosScore, NegDistancePosBodySize, \
	NegDistanceNegSelfCollisionPosBodySize, NegDistanceNegBorderCollisionPosBodySize, \
	ZeroTravelPosScore
from src.snake.state import DirectionalDimensionScoreState, \
	BoardDimensionScoreState, BoardState, BoardScoreState, BoardDimensionState
from src.util.util import filter_models_with_rewards_by_state, get_state_class_from_string, get_reward_class_from_string


def get_test_seed():
	return [1337]


def get_seeds():
	return [123, 234, 345]


def get_states():
	return [BoardDimensionScoreState,
			DirectionalDimensionScoreState]


def get_rewards():
	return [NegTravelPosScore,
			ZeroTravelPosScore,
			PosTravelPosScore,
			NegDistancePosBodySize,
			NegDistanceNegSelfCollisionPosBodySize,
			NegDistanceNegBorderCollisionPosBodySize]


def get_experiment():
	experiment = setup_experiment(QLearning, "reward", get_seeds()[0])
	experiment.train_episodes = 1000
	return experiment


def get_state_from_file_path(file_path):
	return file_path.split("/")[-1].split("_")[0]


def get_reward_from_file_path(file_path):
	return file_path.split("/")[-1].split("_")[1]


def train_experiment(experiment, states=None, rewards=None, pairwise=False):
	def _run_experiment():
		experiment.load()
		experiment.train()
		experiment.save()

	if states is None and rewards is None:
		_run_experiment()
	else:
		if pairwise:
			for state, reward in zip(states, rewards):
				experiment.state = state
				experiment.reward = reward
				_run_experiment()
		else:
			for state in states:
				for reward in rewards:
					experiment.state = state
					experiment.reward = reward
					_run_experiment()


def test_experiment(experiment, n_episodes,
					states=None, rewards=None, pairwise=False):
	def _run_experiment():
		experiment.load()
		return experiment.run(n_episodes)

	score_food_dict = {}

	if states is None and rewards is None:
		scores, foods = _run_experiment()
		score_food_dict[(experiment.state, experiment.reward)] = \
			{"score": scores,
			 "food": foods}
		return score_food_dict

	if pairwise:
		for state, reward in zip(states, rewards):
			experiment.state = state
			experiment.reward = reward
			scores, foods = _run_experiment()
			score_food_dict[(state, reward)] = {"score": scores,
												"food": foods}
	else:
		for state in states:
			for reward in rewards:
				experiment.state = state
				experiment.reward = reward
				scores, foods = _run_experiment()
				score_food_dict[(state, reward)] = {"score": scores,
													"food": foods}
	return score_food_dict


def run_experiment_in_simulator(experiment, state=None, reward=None):
	if state is not None:
		experiment.state = state

	if reward is not None:
		experiment.reward = reward

	experiment.load()

	env = experiment.env
	params = env.params
	agent = SnakeAgent(policy=GreedyPolicy(env),
					   action_value_function=experiment.model_.Q)
	start_app(env, agent, params)


def get_models_from_experiment(experiment, states,
							   rewards, pairwise=False,
							   train_if_missing=False):
	models_dict = {}

	if pairwise:
		for state, reward in zip(states, rewards):
			experiment.state = state
			experiment.reward = reward
			experiment.load()

			if not experiment.has_cached_model() and train_if_missing:
				experiment.train()

			models_dict[experiment.get_model_path()] = experiment.model_
	else:
		for state in states:
			for reward in rewards:
				experiment.state = state
				experiment.reward = reward
				experiment.load()

				if not experiment.has_cached_model() and train_if_missing:
					experiment.train()

				models_dict[experiment.get_model_path()] = experiment.model_

	models = []
	states = []
	rewards = []
	file_paths = []

	for file_path, model in models_dict.items():
		state = get_state_from_file_path(file_path)
		reward = get_reward_from_file_path(file_path)

		models.append(model)
		states.append(state)
		rewards.append(reward)
		file_paths.append(file_path)

	return models, states, rewards, file_paths


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
											state.startswith("Directional"))

	rewards_per_episode = []
	game_score_over_time = []

	for model in models:
		rewards_per_episode.append(model.rewards_per_episode[::10])
		game_score_over_time.append(experiment.env.game_score_coef * model.food_count_per_episode[::10])

	x = np.arange(1, (len(rewards_per_episode[0]) * 10) + 1, 10)

	plt.subplot(211)
	plot_multi_average_reward_over_time(x, rewards_per_episode, rewards)

	plt.subplot(212)
	plot_multi_average_food_count_over_time(x, game_score_over_time, rewards)

	plt.show()


def experiment_board_state_with_and_without_extra_information():
	experiment = get_experiment()
	states = [BoardState,
			  BoardScoreState,
			  BoardDimensionState,
			  BoardDimensionScoreState]
	rewards = [ZeroTravelPosScore]

	models, states, rewards, _ = get_models_from_experiment(experiment,
															states,
															rewards,
															train_if_missing=True)

	rewards_per_episode = []
	food_count_per_episode = []

	for model in models:
		rewards_per_episode.append(model.rewards_per_episode[::10])
		food_count_per_episode.append(model.food_count_per_episode[::10])

	x = np.arange(1, (len(rewards_per_episode[0]) * 10) + 1, 10)

	plt.subplot(211)
	plot_multi_average_reward_over_time(x, rewards_per_episode, states)

	plt.subplot(212)
	plot_multi_average_food_count_over_time(x, food_count_per_episode, states)

	plt.show()

	states = list(map(get_state_class_from_string, np.unique(states)))
	rewards = list(map(get_reward_class_from_string, np.unique(rewards)))
	n_episodes = 1000

	test_results = test_experiment(experiment, n_episodes=n_episodes,
								   states=states, rewards=rewards)

	print("Test Performance on %i episodes: " % (n_episodes,))
	for (state, reward), result in test_results.items():
		print("%s: Mean Score %0.03f, Std Score %0.03f" %
			  (state.__name__,
			   np.mean(result["food"]),
			   np.std(result["food"])), )


def main():
	# train_models()
	# compare_models()
	experiment_board_state_with_and_without_extra_information()
	return

	experiment = setup_experiment(QLearning, "reward", get_seeds()[0])
	experiment.train_episodes = 10000

	state = DirectionalDimensionScoreState
	reward = NegTravelPosScore

	experiment.state = state
	experiment.reward = reward
	experiment.train_episodes = 10000

	if experiment.has_cached_model():
		print("Model is cached!")
	else:
		print("Model is not cached!")
		train_experiment(experiment)

	experiment.load()

	result = test_experiment(experiment, 10000)

	for (state, reward), value in result.items():
		print(state.__name__)
		print(reward.__name__)
		print(np.mean(value["food"]))

	run_experiment_in_simulator(experiment)


if __name__ == '__main__':
	main()
