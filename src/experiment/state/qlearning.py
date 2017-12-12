from src.algorithms.qlearning import QLearning
from src.core.discount_factor import StaticDiscountFactor
from src.core.learning_rate import StaticLearningRate
from src.core.policy import EpsilonGreedyPolicy
from src.experiment.params import ExperimentParameters
from src.experiment.state.analysis import analyze_models
from src.experiment.state.params import get_state_seeds, get_state_reward
from src.experiment.state.train import train_models
from src.snake.environment import SnakeEnvironment
from src.snake.parameters import SnakeParameters


def train():
	params = SnakeParameters()
	params.discount_factor = StaticDiscountFactor(0.85)
	params.learning_rate = StaticLearningRate(0.85)
	params.reward = get_state_reward()

	env = SnakeEnvironment(params)
	params.policy = EpsilonGreedyPolicy(env, params.epsilon)

	exp_params = ExperimentParameters()
	exp_params.env = env
	exp_params.model_class = QLearning
	exp_params.model_params = params

	for seed in get_state_seeds():
		exp_params.seed = seed

		output_dir = "../../../models/qlearning/state/%i" % exp_params.seed
		exp_params.model_output_dir = output_dir

		train_models(exp_params)


def analyze():
	exp_params = ExperimentParameters()

	model_output_dir = "../../../models/qlearning/state/%i" % exp_params.seed
	exp_params.model_output_dir = model_output_dir

	image_output_dir = "../../../images/qlearning/state/%i" % exp_params.seed
	exp_params.image_output_dir = image_output_dir

	analyze_models(exp_params)


if __name__ == "__main__":
	train()
	# analyze()
