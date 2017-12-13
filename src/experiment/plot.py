import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from src.util.math import compute_mean_over_time


def plot_rewards_over_time(learner):
	rewards = learner.rewards_per_episode

	plt.semilogx(rewards)
	plt.title("Reward Over Time")
	plt.xlabel("# of episodes")
	plt.ylabel("Reward")


def plot_average_reward_over_time(learner):
	rewards_over_time = learner.rewards_per_episode
	mean_over_time = compute_mean_over_time(rewards_over_time)

	plt.semilogx(mean_over_time)
	plt.title("Average Reward Over Time")
	plt.xlabel("# of episodes")
	plt.ylabel("Avg. Reward")


def plot_actions_over_time(learner):
	actions = learner.actions_per_episode

	plt.semilogx(actions)
	plt.title("# of Actions Over Time")
	plt.xlabel("# of episodes")
	plt.ylabel("# of actions")


def plot_average_actions_over_time(learner):
	actions = learner.actions_per_episode
	average_actions = compute_mean_over_time(actions)

	plt.semilogx(average_actions)
	plt.title("Average # of Actions Over Time")
	plt.xlabel("# of episodes")
	plt.ylabel("Avg. # of actions")


def plot_exploration_vs_exploitation_over_time(learner):
	actions = learner.actions_per_episode
	exploratory_action = learner.exploratory_actions_per_episode

	plt.semilogx(exploratory_action / actions)
	plt.title("Exploration vs. Exploitation")
	plt.xlabel("# of episodes")
	plt.ylabel("Exploration (%)")


def plot_multi_average_reward_over_time(x, rewards_per_episode, labels):
	mean_reward_over_time = [compute_mean_over_time(rpe)
							 for rpe in rewards_per_episode]

	for label, mrot in zip(labels, mean_reward_over_time):
		plt.semilogx(x, mrot, label=label)

	plt.title("Average Reward Over Time")
	plt.xlabel("# of episodes")
	plt.ylabel("Reward")
	plt.legend(fontsize='x-small')


def plot_multi_average_actions_over_time(x, actions_per_episode, labels):
	average_actions_over_time = [compute_mean_over_time(ape)
								 for ape in actions_per_episode]

	for label, aaot in zip(labels, average_actions_over_time):
		plt.semilogx(x, aaot, label=label)

	plt.title("Average # of Actions Over Time")
	plt.xlabel("# of episodes")
	plt.ylabel("# of actions")
	plt.legend(fontsize='x-small')


def plot_multi_average_food_count_over_time(x, food_count_per_episode, labels):
	average_food_count_over_time = [compute_mean_over_time(fcpe)
									for fcpe in food_count_per_episode]

	for label, afcot in zip(labels, average_food_count_over_time):
		plt.semilogx(x, afcot, label=label)

	plt.title("Average Food Count Over Time")
	plt.xlabel("# of episodes")
	plt.ylabel("Food Count")
	plt.legend(fontsize='x-small')


def plot_multi_average_self_collision_death_over_time(x, self_collision_death_per_episode, labels):
	average_self_collision_death_over_time = [compute_mean_over_time(scdpe)
											  for scdpe in self_collision_death_per_episode]

	for label, ascdot in zip(labels, average_self_collision_death_over_time):
		plt.semilogx(x, ascdot, label=label)

	plt.title("Average Self-Collision Death Over Time")
	plt.xlabel("# of episodes")
	plt.ylabel("Self-Collision Death")
	plt.legend(fontsize='x-small')
