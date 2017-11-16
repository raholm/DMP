import os
from itertools import zip_longest


def grouper(iterable, n, fillvalue=None):
	args = [iter(iterable)] * n
	return zip_longest(*args, fillvalue=fillvalue)


def create_document(images):
	prefix = \
		"""
\\documentclass{article}
		
\\usepackage{graphicx}
\\usepackage{subcaption}
\\usepackage{float}
\\usepackage[font={footnotesize}]{caption}

		
\\begin{document}
		"""

	postfix = \
		"""
\\end{document}
		"""

	image_graphic = \
		"""		
		\\begin{figure}[!htb]
			\\centering
			\\begin{subfigure}[t]{0.475\\textwidth}
				\\centering
				\\includegraphics[height=0.2\\textheight, width=\\textwidth]{%s}
			\\end{subfigure}
			\\hfill
			\\begin{subfigure}[t]{0.475\\textwidth}  
				\\centering 
				\\includegraphics[height=0.2\\textheight, width=\\textwidth]{%s}
			\\end{subfigure}
			\\vskip\\baselineskip
			\\begin{subfigure}[t]{0.475\\textwidth}   
				\\centering 
				\\includegraphics[height=0.2\\textheight, width=\\textwidth]{%s}
			\\end{subfigure}
			\\quad
			\\begin{subfigure}[t]{0.475\\textwidth}   
				\\centering 
				\\includegraphics[height=0.2\\textheight, width=\\textwidth]{%s} 
			\\end{subfigure}
			\\caption{%s : %s}
		\\end{figure}
		"""

	text = prefix
	get_algorithm = lambda image: image.split("/")[2].replace("_", " ")
	get_caption = lambda image: " ".join(image.split("/")[-1].split(".")[0].split("_")[:-4])

	for image1, image2, image3, image4 in grouper(images, 4):
		text += image_graphic % (image1, image2, image3, image4,
								 get_algorithm(image1), get_caption(image1))

	text += postfix

	return text


def get_plot_files(model, type, seed):
	pattern = "/" + "/".join((model, type, seed)) + "/"

	for root, dirs, files in os.walk("../images", topdown=False):
		for file in files:
			path = os.path.join(root, file)

			if pattern in path:
				yield path


def create_state_plots_file():
	models = ["qlearning", "sarsa", "expected_sarsa"]

	images = []

	for model in models:
		images += list(get_plot_files(model, "state", "123"))

	images.sort()

	document = create_document(images)

	with open("state_plots.tex", "w") as outfile:
		outfile.write(document)


def create_reward_plots_file():
	models = ["qlearning", "sarsa", "expected_sarsa"]

	images = []

	for model in models:
		images += list(get_plot_files(model, "reward", "12345"))

	images.sort()

	document = create_document(images)

	with open("reward_plots.tex", "w") as outfile:
		outfile.write(document)


def main():
	# create_state_plots_file()
	create_reward_plots_file()


if __name__ == "__main__":
	main()
