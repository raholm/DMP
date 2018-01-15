from src.algorithms.qlearning import QLearning
from src.core.discount_factor import StaticDiscountFactor
from src.core.learning_rate import StaticLearningRate
from src.core.policy import EpsilonGreedyPolicy
from src.experiments.analysis import get_aggregated_models
from src.experiments.params import ExperimentParameters
from src.experiments.reward.analysis import analyze_models, analyze_aggregated_models
from src.experiments.reward.params import get_reward_states, get_reward_seeds
from src.experiments.reward.train import train_models
from src.snake.environment import SnakeEnvironment
from src.snake.parameters import SnakeParameters


def train():
	params = SnakeParameters()
	params.discount_factor = StaticDiscountFactor(0.85)
	params.learning_rate = StaticLearningRate(0.85)

	env = SnakeEnvironment(params)
	params.policy = EpsilonGreedyPolicy(env, params.epsilon)

	exp_params = ExperimentParameters()
	exp_params.env = env
	exp_params.model_class = QLearning
	exp_params.model_params = params

	for seed in get_reward_seeds():
		exp_params.seed = seed

		output_dir = "../../../models/qlearning/reward/%i" % exp_params.seed
		exp_params.model_output_dir = output_dir

		for state in get_reward_states():
			exp_params.model_params.state = state
			train_models(exp_params)


def analyze():
	exp_params = ExperimentParameters()
	exp_params.seed = get_reward_seeds()[2]

	model_output_dir = "../../../models/qlearning/reward/%i" % exp_params.seed
	exp_params.model_output_dir = model_output_dir

	image_output_dir = "../../../images/qlearning/reward/%i" % exp_params.seed
	exp_params.image_output_dir = image_output_dir

	analyze_models(exp_params)


def analyze_aggregated():
	exp_params = ExperimentParameters()
	exp_params.seed = get_reward_seeds()[0]

	image_output_dir = "../../../images/qlearning/reward/%i" % exp_params.seed
	exp_params.image_output_dir = image_output_dir

	aggregated_models = get_aggregated_models("qlearning", "reward", exp_params, get_reward_seeds())

	filenames = list(aggregated_models.keys())
	models = list(aggregated_models.values())

	analyze_aggregated_models(filenames, models, exp_params)


if __name__ == "__main__":
	# train()
	# analyze()
	analyze_aggregated()
	# analyze_aggregated_reward_food_count_correlations("qlearning")
