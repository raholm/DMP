class Environment(object):
	def start_new_episode(self):
		"""
		Starts a completely new episode.

		Returns
		-------
		state : State
		    The initial state.
		"""
		pass

	def episode_is_done(self):
		"""
		Checks if the current episode has ended. i.e.,
		the agent is in a terminal state.

		Returns
		-------
		status : bool
		    True if agent is in terminal state, False otherwise.
		"""
		pass

	def step(self, action):
		"""
		Takes a single step in the environment of the current episode by
		performing the given action.

		Parameters
		----------
		action : Action
		    The action to perform.

		Returns
		-------
		state : State
		    The new state after performing the given the action in the current state.

		reward : Reward
		    The reward for performing the given action in the current state.
		"""
		pass

	def get_valid_actions(self, state):
		"""
		Returns the actions that can be performed in the given state.

		Parameters
		----------
		state : State
		    The state.

		Returns
		-------
		actions : List of Action
		    A list of valid actions.
		"""

	def get_state_reward(self, state, action):
		"""
		Returns the next state-reward pair given a state and an action.

		Parameters
		----------
		state : State
		    The state.

		action : Action
		    The action.

		Returns
		-------
		state : State
		    The new state after performing the given action in the given state.

		reward : Reward
		    The reward for performing the given action in the given state.
		"""
		pass
