import os


class ExperimentParameters(object):
	def __init__(self):
		self.env = None

		self.model_class = None
		self.model_params = None
		self._model_output_dir = None

		self.seed = 123

	@property
	def model_output_dir(self):
		return self._model_output_dir

	@model_output_dir.setter
	def model_output_dir(self, value):
		self._model_output_dir = value

		if not os.path.exists(self.model_output_dir):
			os.makedirs(self.model_output_dir)
