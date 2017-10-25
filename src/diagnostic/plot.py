from matplotlib import pyplot as plt

from src.util.math import compute_mean_over_time


def diagnostic_plot_exploration_vs_exploitation_over_time(learner):
	actions = learner.actions_per_episode
	exploratory_action = learner.exploratory_actions_per_episode
	exploration = compute_mean_over_time(exploratory_action / actions)

	plt.semilogx(exploration)
	plt.title("Exploration vs. Exploitation")
	plt.xlabel("# of episodes")
	plt.ylabel("Exploration (%)")