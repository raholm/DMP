from src.snake.state import BoardDimensionScoreState, DirectionalDistanceDimensionScoreState


def get_reward_seeds():
	return [234, 345, 456]


def get_reward_states():
	return [BoardDimensionScoreState, DirectionalDistanceDimensionScoreState]
