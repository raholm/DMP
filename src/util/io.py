import pickle


def write_model(model, outfile):
	with open(outfile, "wb") as ofile:
		pickle.dump(model, ofile)


def read_model(infile):
	with open(infile, "rb") as ifile:
		model = pickle.load(ifile)

	return model
