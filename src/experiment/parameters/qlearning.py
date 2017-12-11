from src.algorithms.qlearning import QLearning
from src.core.policy import EpsilonGreedyPolicy
from src.experiment.parameters.analysis import analyze_models
from src.experiment.parameters.params import get_snake_parameters
from src.experiment.train import train_and_store_model
from src.snake.environment import SnakeEnvironment
from src.experiment.params import ExperimentParameters


def train():
	for params in get_snake_parameters():
		env = SnakeEnvironment(params)
		params.policy = EpsilonGreedyPolicy(env, params.epsilon)

		exp_params = ExperimentParameters()
		exp_params.env = env
		exp_params.model_class = QLearning
		exp_params.model_params = params
		exp_params.seed = 666

		output_dir = "../../../models/qlearning/params/%i" % exp_params.seed
		exp_params.model_output_dir = output_dir

		train_and_store_model(exp_params)


def analyze():
	exp_params = ExperimentParameters()
	exp_params.seed = 666

	model_output_dir = "../../../models/qlearning/params/%i" % exp_params.seed
	exp_params.model_output_dir = model_output_dir

	image_output_dir = "../../../images/qlearning/params/%i" % exp_params.seed
	exp_params.image_output_dir = image_output_dir

	analyze_models(exp_params)


def main():
	train()
	analyze()


if __name__ == '__main__':
	main()
