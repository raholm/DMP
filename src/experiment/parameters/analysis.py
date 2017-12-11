from src.experiment.analysis import read_models, plot_model_analysis


def analyze_models(params):
	models, filenames = read_models(params)

	labels = []

	for filename in filenames:
		parts = filename.split("_")
		label = "g=%.2f,a=%.2f" % (float(parts[-4]), float(parts[-3]),)
		labels.append(label)

	plot_model_analysis(models, labels, "params", params)
