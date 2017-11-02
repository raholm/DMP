from src.algorithms.qlearning import QLearning
from src.core.policy import EpsilonGreedyPolicy
from src.experiment.parameters import ExperimentParameters
from src.experiment.state.analysis import analyze_models
from src.experiment.state.train import train_models
from src.snake.environment import SnakeEnvironment
from src.snake.parameters import SnakeParameters


def train():
	params = SnakeParameters()
	env = SnakeEnvironment(params)
	params.policy = EpsilonGreedyPolicy(env, params.epsilon)
	output_dir = "../../../models/qlearning/state"

	exp_params = ExperimentParameters()
	exp_params.env = env
	exp_params.model_class = QLearning
	exp_params.model_params = params
	exp_params.model_output_dir = output_dir

	train_models(exp_params)


def analyze():
	output_dir = "../../../models/qlearning/state"

	exp_params = ExperimentParameters()
	exp_params.model_output_dir = output_dir

	analyze_models(exp_params)


if __name__ == "__main__":
	train()
	analyze()
