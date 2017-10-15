class DiscountFactor(object):
	def __call__(self, state):
		raise NotImplementedError


class StaticDiscountFactor(DiscountFactor):
	def __init__(self, discount):
		self.discount = discount

	def __call__(self, state):
		return self.discount
