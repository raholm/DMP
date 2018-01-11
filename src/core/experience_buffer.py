import random
import numpy as np


class ExperienceBuffer():
	def __init__(self, buffer_size=10):
		self.buffer = []
		self.buffer_size = buffer_size

	def add(self, experience):
		experience = np.reshape(np.array(experience), [1, 5])

		if len(self.buffer) + len(experience) >= self.buffer_size:
			self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []

		self.buffer.extend(experience)

	def sample(self, size):
		idx = np.random.choice(len(self.buffer), size)
		return np.reshape(np.array([self.buffer[i] for i in idx]), [size, 5])
