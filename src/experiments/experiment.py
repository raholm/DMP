import os

from src.core.reward import Reward
from src.core.state import State
from src.util.io import write_model, read_model, get_model_path


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
		if not isinstance(state, State):
			raise ValueError("Input is not a state : %s" % (type(state),))

		self.env.params.state = state

	@property
	def reward(self):
		return self.env.params.reward

	@reward.setter
	def reward(self, reward):
		if not isinstance(reward, Reward):
			raise ValueError("Input is not a reward : %s" % (type(reward),))

		self.env.params.reward = reward

	def train(self):
		self._check_model_is_loaded()
		self.model_ = self.model_.train(self.env, self.train_episodes)
		return self

	def run(self, n_episodes):
		self._check_model_is_loaded()
		return self.model_.run(self.env, n_episodes)

	def load(self):
		self._load_or_create_model()

	def save(self):
		self._check_model_is_loaded()

		if not self._model_exists():
			write_model(self.model_, self._get_model_path())

	def _check_model_is_loaded(self):
		if self.model_ is None:
			raise ValueError("The model has not been loaded. Call load() first.")

	def _load_or_create_model(self):
		if self._model_exists():
			self.model_ = read_model(self._get_model_path())
		else:
			self.model_ = self.model_class(
				action_value_function=self.value_function,
				policy=self.policy,
				learning_rate=self.learning_rate,
				discount_factor=self.discount_factor)

	def _model_exists(self):
		if os.path.isfile(self._get_model_path()):
			return True
		return False

	def _get_model_path(self):
		return os.path.join(get_model_path(self.model_class),
							self._get_model_filename())

	def _get_model_filename(self):
		return "%s_%s_%s_%s_%s_%i_%.2f_%.2f_%.2f_%ix%i" % (
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
