from src.snake.reward import ZeroTravelPosScore


def get_state_seeds():
	return [234, 345, 456][0:2]


def get_state_reward():
	return ZeroTravelPosScore
