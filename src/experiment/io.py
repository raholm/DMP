import os

from src.util.io import read_model


def read_models(params):
	filenames = []
	models = []

	for subdir, dirs, files in os.walk(params.model_output_dir):
		for file in files:
			file_path = os.path.join(subdir, file)
			filenames.append(file)
			models.append(read_model(file_path))

	return models, filenames