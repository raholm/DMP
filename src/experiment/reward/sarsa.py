from src.algorithms.sarsa import Sarsa
from src.core.policy import EpsilonGreedyPolicy
from src.experiment.parameters import ExperimentParameters
from src.experiment.reward.train import train_models
from src.snake.environment import SnakeEnvironment
from src.snake.parameters import SnakeParameters


def train():
	params = SnakeParameters()
	env = SnakeEnvironment(params)
	params.policy = EpsilonGreedyPolicy(env, params.epsilon)

	exp_params = ExperimentParameters()
	exp_params.env = env
	exp_params.model_class = Sarsa
	exp_params.model_params = params

	output_dir = "../../../models/sarsa/reward/%i" % exp_params.seed
	exp_params.model_output_dir = output_dir

	# Loop over states

	train_models(exp_params)


def analyze():
	exp_params = ExperimentParameters()

	output_dir = "../../../models/sarsa/reward/%i" % exp_params.seed
	exp_params.model_output_dir = output_dir


# analyze_models(exp_params)


if __name__ == "__main__":
	train()
# analyze()
