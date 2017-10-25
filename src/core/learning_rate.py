from collections import defaultdict


class LearningRate(object):
	def __call__(self, state):
		raise NotImplementedError


class StaticLearningRate(LearningRate):
	def __init__(self, rate):
		self.rate = rate

	def __call__(self, state):
		return self.rate


class FrequencyLearningRate(LearningRate):
	def __init__(self):
		self.frequencies = defaultdict(int)

	def __call__(self, state):
		self.frequencies[state] += 1
		return 1 / self.frequencies[state]
