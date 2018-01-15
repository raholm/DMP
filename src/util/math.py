import numpy as np

from scipy import spatial


def manhattan(x, y):
	return spatial.distance.cityblock(x, y)


def compute_mean_over_time(x):
	mean_over_time = np.zeros(len(x))
	mean_over_time[0] = x[0]

	for t in range(1, len(mean_over_time)):
		previous_mean = mean_over_time[t - 1]
		mean_over_time[t] = previous_mean + (x[t] - previous_mean) / t

	return mean_over_time


def compute_correlation(xs, ys):
	def get_corr_coefs(models):
		corr_coefs = []

		for x, y in zip(xs, ys):
			if np.all(x) == 0 or np.all(y) == 0:
				continue

			corr_coef = np.corrcoef(x, y)
			corr_coefs.append(corr_coef)

		return corr_coefs
