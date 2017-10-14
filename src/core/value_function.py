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
		pass

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
		pass


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
		pass

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
		pass
