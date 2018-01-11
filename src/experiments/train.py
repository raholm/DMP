import os
from timeit import default_timer as timer

from src.util.io import write_model


def train_and_store_model(params):
	learner = params.model_class(action_value_function=params.model_params.value_function,
								 policy=params.model_params.policy,
								 learning_rate=params.model_params.learning_rate,
								 discount_factor=params.model_params.discount_factor)

	start = timer()

	learner.train(params.env, params.model_params.train_episodes)

	print("Training time:", timer() - start)

	write_model(learner, os.path.join(params.model_output_dir, "%s.p" % params.model_params.file_str))

	return learner
