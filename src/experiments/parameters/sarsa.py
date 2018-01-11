from src.algorithms.sarsa import Sarsa
from src.core.policy import EpsilonGreedyPolicy
from src.experiments.parameters.analysis import analyze_models
from src.experiments.parameters.params import get_snake_parameters, get_parameters_seed
from src.experiments.train import train_and_store_model
from src.snake.environment import SnakeEnvironment
from src.experiments.params import ExperimentParameters


def train():
	for params in get_snake_parameters():
		env = SnakeEnvironment(params)
		params.policy = EpsilonGreedyPolicy(env, params.epsilon)

		exp_params = ExperimentParameters()
		exp_params.env = env
		exp_params.model_class = Sarsa
		exp_params.model_params = params
		exp_params.seed = get_parameters_seed()

		output_dir = "../../../models/sarsa/params/%i" % exp_params.seed
		exp_params.model_output_dir = output_dir

		train_and_store_model(exp_params)


def analyze():
	exp_params = ExperimentParameters()
	exp_params.seed = get_parameters_seed()

	model_output_dir = "../../../models/sarsa/params/%i" % exp_params.seed
	exp_params.model_output_dir = model_output_dir

	image_output_dir = "../../../images/sarsa/params/%i" % exp_params.seed
	exp_params.image_output_dir = image_output_dir

	analyze_models(exp_params)


def main():
	train()
	analyze()


if __name__ == '__main__':
	main()
