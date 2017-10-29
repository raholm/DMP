import pickle


def write_learner(learner, outfile):
	pickle.dump(learner, open(outfile, "wb"))


def read_learner(infile):
	return pickle.load(open(infile, "rb"))
