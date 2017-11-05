from src.core.discount_factor import StaticDiscountFactor
from src.core.learning_rate import StaticLearningRate
from src.core.value_function import DictActionValueFunction
from src.snake.reward import NegativeTravelPositiveFood
from src.snake.snake import SnakeDirection
from src.snake.state import SnakeFoodScoreState


class SnakeParameters(object):
	def __init__(self):
		# Game Related
		self.update_rate = 250

		# Board Related
		self.rows = 5
		self.cols = 5
		self.cell_size = 48

		# Snake Related
		self.initial_snake_size = 1
		self.initial_snake_position = (2, 2)
		self.initial_snake_direction = SnakeDirection.East
		self.tail_size_increase = 1

		# Learning Related
		self.state = SnakeFoodScoreState
		self.reward = NegativeTravelPositiveFood
		self.discount_factor = StaticDiscountFactor(0.85)
		self.learning_rate = StaticLearningRate(0.15)
		self.epsilon = 0.15
		self.policy = None
		self.value_function = DictActionValueFunction(0)
		self.train_episodes = 1000000

	@property
	def file_str(self):
		return "%s_%s_%s_%s_%s_%i_%.2f_%ix%i" % (self.state.__name__,
												 self.reward.__name__,
												 self.policy.__class__.__name__,
												 self.discount_factor.__class__.__name__,
												 self.learning_rate.__class__.__name__,
												 self.train_episodes,
												 self.epsilon,
												 self.rows, self.cols)
