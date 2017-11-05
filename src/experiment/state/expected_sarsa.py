from src.algorithms.sarsa import ExpectedSarsa
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

	exp_params = ExperimentParameters()
	exp_params.env = env
	exp_params.model_class = ExpectedSarsa
	exp_params.model_params = params

	output_dir = "../../../models/expected_sarsa/state/%i" % exp_params.seed
	exp_params.model_output_dir = output_dir

	train_models(exp_params)


def analyze():
	exp_params = ExperimentParameters()

	output_dir = "../../../models/expected_sarsa/state/%i" % exp_params.seed
	exp_params.model_output_dir = output_dir

	analyze_models(exp_params)


if __name__ == "__main__":
	train()
	# analyze()
