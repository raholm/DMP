import matplotlib.pyplot as plt

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
