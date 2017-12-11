from src.algorithms.sarsa import Sarsa
from src.core.policy import EpsilonGreedyPolicy
from src.experiment.params import ExperimentParameters
from src.experiment.reward.analysis import analyze_models
from src.experiment.reward.params import get_reward_states, get_reward_seeds
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

	for seed in get_reward_seeds():
		exp_params.seed = seed

		output_dir = "../../../models/sarsa/reward/%i" % exp_params.seed
		exp_params.model_output_dir = output_dir

		for state in get_reward_states():
			exp_params.model_params.state = state
			train_models(exp_params)


def analyze():
	exp_params = ExperimentParameters()

	model_output_dir = "../../../models/sarsa/reward/%i" % exp_params.seed
	exp_params.model_output_dir = model_output_dir

	image_output_dir = "../../../images/sarsa/reward/%i" % exp_params.seed
	exp_params.image_output_dir = image_output_dir

	analyze_models(exp_params)


if __name__ == "__main__":
	train()
	# analyze()
