import importlib
import os


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


def get_state_class_from_string(state):
	return getattr(importlib.import_module("src.snake.state"), state)


def get_reward_class_from_string(reward):
	return getattr(importlib.import_module("src.snake.reward"), reward)


def get_state_from_file_path(file_path):
	return file_path.split("/")[-1].split("_")[0]


def get_reward_from_file_path(file_path):
	return file_path.split("/")[-1].split("_")[1]


def create_dir(directory):
	if not os.path.isdir(directory):
		os.makedirs(directory)
