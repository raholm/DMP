def filter_models_by_state(models, states, predicate):
	current_models = []
	current_states = []

	for i, state in enumerate(states):
		if predicate(state):
			current_models.append(models[i])
			current_states.append(state)

	return current_models, current_states


def filter_models_with_rewards_by_state(models, states, rewards, predicate):
	current_models = []
	current_states = []
	current_rewards = []

	for i, (state, reward) in enumerate(zip(states, rewards)):
		if predicate(state):
			current_models.append(models[i])
			current_states.append(state)
			current_rewards.append(reward)

	return current_models, current_states, current_rewards
