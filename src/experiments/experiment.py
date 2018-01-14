import os
import numpy as np

from src.algorithms.qlearning import QLearning
from src.algorithms.sarsa import Sarsa, ExpectedSarsa
from src.core.discount_factor import StaticDiscountFactor
from src.core.learning_rate import StaticLearningRate
from src.core.policy import EpsilonGreedyPolicy, GreedyPolicy

from src.core.reward import Reward
from src.core.state import State
from src.core.value_function import DictActionValueFunction
from src.main import start_app
from src.snake.agent import SnakeAgent
from src.snake.environment import SnakeEnvironment
from src.snake.parameters import SnakeParameters
from src.util.io import write_model, read_model, get_model_path
from src.util.util import get_state_from_file_path, get_reward_from_file_path, create_dir


class Experiment(object):
	def __init__(self, env=None,
				 model_class=None,
				 experiment_type=None,
				 discount_factor=None,
				 learning_rate=None,
				 epsilon=None,
				 policy=None,
				 value_function=None,
				 train_episodes=None,
				 seed=None):
		self.env = env
		self.model_class = model_class
		self.experiment_type = experiment_type
		self.discount_factor = discount_factor
		self.learning_rate = learning_rate
		self.epsilon = epsilon
		self.policy = policy
		self.value_function = value_function
		self.train_episodes = train_episodes
		self.seed = seed

		self.model_ = None

	@property
	def state(self):
		return self.env.params.state

	@state.setter
	def state(self, state):
		if not issubclass(state, State):
			raise ValueError("Input is not a state : %s" % (type(state),))

		self.env.params.state = state

	@property
	def reward(self):
		return self.env.params.reward

	@reward.setter
	def reward(self, reward):
		if not issubclass(reward, Reward):
			raise ValueError("Input is not a reward : %s" % (type(reward),))

		self.env.params.reward = reward

	@property
	def seed(self):
		return self._seed

	@seed.setter
	def seed(self, value):
		self._seed = value
		np.random.seed(self._seed)

	def train(self):
		self._check_model_is_loaded()

		if self._model_exists():
			return self

		self.model_ = self.model_.train(self.env, self.train_episodes)
		return self

	def run(self, n_episodes):
		self._check_model_is_loaded()
		policy = GreedyPolicy(self.env)
		agent = SnakeAgent(policy, self.model_.Q)
		return self.env.run(agent, n_episodes)

	def load(self):
		self._load_or_create_model()

	def save(self):
		self._check_model_is_loaded()

		if not self._model_exists():
			write_model(self.model_, self.get_model_path())

	def has_cached_model(self):
		return self._model_exists()

	def get_model_path(self):
		directory = os.path.join(get_model_path(self.model_class),
								 self.experiment_type,
								 str(self.seed))
		create_dir(directory)
		return os.path.join(directory, self._get_model_filename())

	def _check_model_is_loaded(self):
		if self.model_ is None:
			raise ValueError("The model has not been loaded. Call load() first.")

	def _load_or_create_model(self):
		if self._model_exists():
			self.model_ = read_model(self.get_model_path())
		else:
			self.model_ = self.model_class(
				action_value_function=self.value_function,
				policy=self.policy,
				learning_rate=self.learning_rate,
				discount_factor=self.discount_factor)

	def _model_exists(self):
		if os.path.isfile(self.get_model_path()):
			return True
		return False

	def _get_model_filename(self):
		return "%s_%s_%s_%s_%s_%i_%.2f_%.2f_%.2f_%ix%i.p" % (
			self.state.__name__,
			self.reward.__name__,
			self.policy.__class__.__name__,
			self.discount_factor.__class__.__name__,
			self.learning_rate.__class__.__name__,
			self.train_episodes,
			self.discount_factor.discount,
			self.learning_rate.rate,
			self.epsilon,
			self.env.rows,
			self.env.cols)


def setup_experiment(model_class, exp_type, seed):
	if model_class not in (QLearning, Sarsa, ExpectedSarsa):
		raise ValueError("Unknown model : %s" % (model_class,))

	if model_class == QLearning:
		learning_rate = 0.85
		discount_factor = 0.85
	else:
		learning_rate = 0.15
		discount_factor = 0.85

	epsilon = 0.15
	train_episodes = 1000000
	value_function = DictActionValueFunction(0)
	learning_rate = StaticLearningRate(learning_rate)
	discount_factor = StaticDiscountFactor(discount_factor)

	snake_params = SnakeParameters()
	env = SnakeEnvironment(snake_params)
	policy = EpsilonGreedyPolicy(env, epsilon)

	experiment = Experiment(env=env,
							model_class=model_class,
							experiment_type=exp_type,
							seed=seed,
							epsilon=epsilon,
							train_episodes=train_episodes,
							value_function=value_function,
							learning_rate=learning_rate,
							discount_factor=discount_factor,
							policy=policy)
	experiment.load()
	return experiment


def train_experiment(experiment, states=None, rewards=None, pairwise=False):
	def _run_experiment():
		experiment.load()
		experiment.train()
		experiment.save()

	if states is None and rewards is None:
		_run_experiment()
	else:
		if pairwise:
			for state, reward in zip(states, rewards):
				experiment.state = state
				experiment.reward = reward
				_run_experiment()
		else:
			for state in states:
				for reward in rewards:
					experiment.state = state
					experiment.reward = reward
					_run_experiment()


def test_experiment(experiment, n_episodes,
					states=None, rewards=None, pairwise=False):
	def _run_experiment():
		experiment.load()
		return experiment.run(n_episodes)

	scores_dict = {}

	if states is None and rewards is None:
		scores = _run_experiment()
		scores_dict[(experiment.state, experiment.reward)] = scores
		return scores_dict

	if pairwise:
		for state, reward in zip(states, rewards):
			experiment.state = state
			experiment.reward = reward
			scores = _run_experiment()
			scores_dict[(state, reward)] = scores
	else:
		for state in states:
			for reward in rewards:
				experiment.state = state
				experiment.reward = reward
				scores = _run_experiment()
				scores_dict[(state, reward)] = scores

	return scores_dict


def run_experiment_in_simulator(experiment, state=None, reward=None):
	if state is not None:
		experiment.state = state

	if reward is not None:
		experiment.reward = reward

	experiment.load()

	env = experiment.env
	params = env.params
	agent = SnakeAgent(policy=GreedyPolicy(env),
					   action_value_function=experiment.model_.Q)
	start_app(env, agent, params)


def get_models_from_experiment(experiment, states,
							   rewards, pairwise=False,
							   train_if_missing=False):
	models_dict = {}

	if pairwise:
		for state, reward in zip(states, rewards):
			experiment.state = state
			experiment.reward = reward
			experiment.load()

			if not experiment.has_cached_model() and train_if_missing:
				experiment.train()
				experiment.save()

			models_dict[experiment.get_model_path()] = experiment.model_
	else:
		for state in states:
			for reward in rewards:
				experiment.state = state
				experiment.reward = reward
				experiment.load()

				if not experiment.has_cached_model() and train_if_missing:
					experiment.train()
					experiment.save()

				models_dict[experiment.get_model_path()] = experiment.model_

	models = []
	states = []
	rewards = []
	file_paths = []

	for file_path, model in models_dict.items():
		state = get_state_from_file_path(file_path)
		reward = get_reward_from_file_path(file_path)

		models.append(model)
		states.append(state)
		rewards.append(reward)
		file_paths.append(file_path)

	return models, states, rewards, file_paths
