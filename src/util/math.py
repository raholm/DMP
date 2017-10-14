from scipy import spatial


def manhattan(x, y):
	return spatial.distance.cityblock(x, y)
