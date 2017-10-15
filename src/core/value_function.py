class StateValueFunction(object):
	def get_value(self, state):
		"""
		Gets the current estimated state value.

		Parameters
		----------
		state : State
		    The state.

		Returns
		-------
		value : float
		    The estimated value of the state.
		"""
		raise NotImplementedError

	def set_value(self, state, value):
		"""
		Updates the current state value to the new value.

		Parameters
		----------
		state : State
		    The state.
		value : float
		    The new estimate.

		Returns
		-------
		self
		"""
		raise NotImplementedError


class ActionValueFunction(object):
	def get_value(self, state, action):
		"""
		Gets the current estimated state-action value.

		Parameters
		----------
		state : State
		    The state.
		action : Action
		    The action.

		Returns
		-------
		value : float
		    The estimated value of the state-action pair.
		"""
		raise NotImplementedError

	def set_value(self, state, action, value):
		"""
		Updates the current state-action value to the new value.

		Parameters
		----------
		state : State
		    The state.
		action : Action
		    The action.
		value : float
		    The new estimate.

		Returns
		-------
		self
		"""
		raise NotImplementedError


class DictActionValueFunction(ActionValueFunction):
	def __init__(self, default_value):
		self.values = dict()
		self.default_value = default_value

	def set_value(self, state, action, value):
		if state is None or action is None:
			return

		self.values[(state, action)] = value

	def get_value(self, state, action):
		if state is None or action is None:
			return self.default_value
		value = self.values.get((state, action), self.default_value)
		return value
