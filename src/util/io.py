import os
import pickle

from src.algorithms.qlearning import QLearning
from src.algorithms.sarsa import Sarsa, ExpectedSarsa


def write_model(model, outfile):
	with open(outfile, "wb") as ofile:
		pickle.dump(model, ofile)


def read_model(infile):
	with open(infile, "rb") as ifile:
		model = pickle.load(ifile)

	return model


def get_model_path(model_class_or_type):
	if model_class_or_type == QLearning:
		model_class_or_type = "qlearning"
	elif model_class_or_type == Sarsa:
		model_class_or_type = "sarsa"
	elif model_class_or_type == ExpectedSarsa:
		model_class_or_type = "expected_sarsa"

	if model_class_or_type not in ("qlearning", "sarsa", "expected_sarsa"):
		raise ValueError("Unknown model type : %s" % (model_class_or_type,))

	return os.path.join(get_project_path(), "models", model_class_or_type)


def get_model_file_paths_from_dir(directory):
	file_paths = [os.path.join(directory, file)
				  for file in os.listdir(directory)
				  if file.endswith(".p")]
	return file_paths


def get_project_path():
	return os.environ["DMP_PATH"]
