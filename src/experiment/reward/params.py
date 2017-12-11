from src.snake.state import BoardDimensionScoreState, DirectionalDistanceDimensionScoreState


def get_reward_states():
	return [BoardDimensionScoreState, DirectionalDistanceDimensionScoreState]