import numpy as np
from numpy import array
def svm_read_problem(data_file_name):
	"""
	svm_read_problem(data_file_name) -> [y, x]

	Read LIBSVM-format data from data_file_name and return labels y
	and data instances x.
	"""
	prob_y = []
	# prob_x = []
	prob_x=[]
	for line in open(data_file_name):
		line = line.split(None, 1)
		# In case an instance with all zero features
		if len(line) == 1: line += ['']
		label, features = line
		xi = []
		for e in features.split():
			ind, val = e.split(":")
			xi.append(float(val))
		prob_y += [float(label)]
		prob_x .append(xi)
	return (array(prob_y), array(prob_x))