from src.algorithms.sarsa import ExpectedSarsa
from src.core.discount_factor import StaticDiscountFactor
from src.core.learning_rate import StaticLearningRate
from src.core.policy import EpsilonGreedyPolicy
from src.experiments.analysis import get_aggregated_models
from src.experiments.params import ExperimentParameters
from src.experiments.state.analysis import analyze_models, analyze_aggregated_models
from src.experiments.state.params import get_state_seeds, get_state_reward
from src.experiments.state.train import train_models
from src.snake.environment import SnakeEnvironment
from src.snake.parameters import SnakeParameters


def train():
	params = SnakeParameters()
	params.discount_factor = StaticDiscountFactor(0.95)
	params.learning_rate = StaticLearningRate(0.15)
	params.reward = get_state_reward()

	env = SnakeEnvironment(params)
	params.policy = EpsilonGreedyPolicy(env, params.epsilon)

	exp_params = ExperimentParameters()
	exp_params.env = env
	exp_params.model_class = ExpectedSarsa
	exp_params.model_params = params

	for seed in get_state_seeds():
		exp_params.seed = seed

		output_dir = "../../../models/expected_sarsa/state/%i" % exp_params.seed
		exp_params.model_output_dir = output_dir

		train_models(exp_params)


def analyze():
	exp_params = ExperimentParameters()

	model_output_dir = "../../../models/expected_sarsa/state/%i" % exp_params.seed
	exp_params.model_output_dir = model_output_dir

	image_output_dir = "../../../images/expected_sarsa/state/%i" % exp_params.seed
	exp_params.image_output_dir = image_output_dir

	analyze_models(exp_params)


def analyze_aggregated():
	exp_params = ExperimentParameters()
	exp_params.seed = get_state_seeds()[0]

	image_output_dir = "../../../images/expected_sarsa/state/%i" % exp_params.seed
	exp_params.image_output_dir = image_output_dir

	aggregated_models = get_aggregated_models("expected_sarsa", "state", exp_params, get_state_seeds())

	filenames = list(aggregated_models.keys())
	models = list(aggregated_models.values())

	analyze_aggregated_models(filenames, models, exp_params)


if __name__ == "__main__":
	# train()
	# analyze()
	analyze_aggregated()
