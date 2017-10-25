import numpy as np

from scipy import spatial


def manhattan(x, y):
	return spatial.distance.cityblock(x, y)


def compute_mean_over_time(x):
	mean_over_time = np.zeros(len(x))
	mean_over_time[0] = x[0]

	for t in range(1, len(mean_over_time)):
		previous_mean = mean_over_time[t - 1]
		mean_over_time[t] = previous_mean + (1 / t) * (x[t] - previous_mean)

	return mean_over_time
