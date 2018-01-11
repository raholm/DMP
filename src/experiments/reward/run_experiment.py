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
	BoardDimensionScoreState
from src.util.util import filter_models_with_rewards_by_state


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


def extract_models_from_experiment(experiment, states,
								   rewards, pairwise=False):
	models = {}

	if pairwise:
		for state, reward in zip(states, rewards):
			experiment.state = state
			experiment.reward = reward
			experiment.load()

			if not experiment.has_cached_model():
				experiment.train()

			models[experiment.get_model_path()] = experiment.model_
	else:
		for state in states:
			for reward in rewards:
				experiment.state = state
				experiment.reward = reward
				experiment.load()

				if not experiment.has_cached_model():
					experiment.train()

				models[experiment.get_model_path()] = experiment.model_

	return models


def get_experiment():
	experiment = setup_experiment(QLearning, "reward", get_seeds()[0])
	experiment.train_episodes = 10000
	return experiment


def train_models():
	train_experiment(get_experiment(), get_states(), get_rewards())


def compare_models():
	experiment = setup_experiment(QLearning, "reward", get_seeds()[0])
	experiment.train_episodes = 100000

	filepath_model_dict = extract_models_from_experiment(experiment,
														 get_states(),
														 get_rewards())

	extract_state = lambda file_path: file_path.split("/")[-1].split("_")[0]
	extract_reward = lambda file_path: file_path.split("/")[-1].split("_")[1]

	models = []
	states = []
	rewards = []

	for file_path, model in filepath_model_dict.items():
		state = extract_state(file_path)
		reward = extract_reward(file_path)

		models.append(model)
		states.append(state)
		rewards.append(reward)

	models, states, rewards = \
		filter_models_with_rewards_by_state(models, states, rewards,
											lambda state:
											state.startswith("Directional"))

	rewards_per_episode = []
	food_count_per_episode = []

	for model in models:
		rewards_per_episode.append(model.rewards_per_episode[::10])
		food_count_per_episode.append(model.food_count_per_episode[::10])

	x = np.arange(1, (len(rewards_per_episode[0]) * 10) + 1, 10)

	plt.subplot(211)
	plot_multi_average_reward_over_time(x, rewards_per_episode, rewards)

	plt.subplot(212)
	plot_multi_average_food_count_over_time(x, food_count_per_episode, rewards)

	plt.show()


def main():
	# train_models()
	compare_models()
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
