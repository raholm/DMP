from src.algorithms.qlearning import QLearning
from src.core.policy import EpsilonGreedyPolicy
from src.experiment.parameters import ExperimentParameters
from src.experiment.reward.analysis import analyze_models
from src.experiment.reward.train import train_models
from src.snake.environment import SnakeEnvironment
from src.snake.parameters import SnakeParameters
from src.snake.state import BoardScoreState, SnakeFoodScoreState, DirectionalDistanceScoreState, DirectionalScoreState


def train():
	params = SnakeParameters()
	env = SnakeEnvironment(params)
	params.policy = EpsilonGreedyPolicy(env, params.epsilon)

	exp_params = ExperimentParameters()
	exp_params.env = env
	exp_params.model_class = QLearning
	exp_params.model_params = params

	output_dir = "../../../models/qlearning/reward/%i" % exp_params.seed
	exp_params.model_output_dir = output_dir

	states = [BoardScoreState, SnakeFoodScoreState,
			  DirectionalScoreState, DirectionalDistanceScoreState]

	for state in states:
		exp_params.model_params.state = state
		train_models(exp_params)


def analyze():
	exp_params = ExperimentParameters()

	model_output_dir = "../../../models/qlearning/reward/%i" % exp_params.seed
	exp_params.model_output_dir = model_output_dir

	image_output_dir = "../../../images/qlearning/reward/%i" % exp_params.seed
	exp_params.image_output_dir = image_output_dir

	analyze_models(exp_params)


if __name__ == "__main__":
	train()
	# analyze()
