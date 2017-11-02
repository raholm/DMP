import pickle

from src.experiment.state.analysis import analyze_models

from src.algorithms.sarsa import Sarsa
from src.core.policy import EpsilonGreedyPolicy
from src.experiment.parameters import ExperimentParameters
from src.experiment.state.train import train_models
from src.main import start_app
from src.snake.agent import SnakeAgent
from src.snake.environment import SnakeEnvironment
from src.snake.parameters import SnakeParameters


def run():
	params = SnakeParameters()
	env = SnakeEnvironment(params)
	value_function = pickle.load(open("../../models/sarsa_%s.p" % params.file_str, "rb"))

	agent = SnakeAgent(policy=EpsilonGreedyPolicy(env, 0),
					   action_value_function=value_function)

	start_app(env, agent, params)


def train():
	params = SnakeParameters()
	env = SnakeEnvironment(params)
	params.policy = EpsilonGreedyPolicy(env, params.epsilon)
	output_dir = "../../../models/sarsa/state"

	exp_params = ExperimentParameters()
	exp_params.env = env
	exp_params.model_class = Sarsa
	exp_params.model_params = params
	exp_params.model_output_dir = output_dir

	train_models(exp_params)


def analyze():
	output_dir = "../../../models/sarsa/state"

	exp_params = ExperimentParameters()
	exp_params.model_output_dir = output_dir

	analyze_models(exp_params)


if __name__ == "__main__":
	train_models()
# analyze_models()
