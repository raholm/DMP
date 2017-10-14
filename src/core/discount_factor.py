class DiscountFactor(object):
	def discount(self, state):
		pass


class StaticDiscountFactor(DiscountFactor):
	def __init__(self, discount):
		self.discount = discount

	def discount(self, state):
		return self.discount
