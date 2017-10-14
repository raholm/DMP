class Policy(object):
	def __init__(self, actions):
		self.actions = actions


class EpsilonGreedyPolicy(object):
	def __init__(self, actions, epsilon):
		super.__init__(actions)

		self.epsilon = epsilon
