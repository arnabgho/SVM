import numpy as np
from chunking import *
from parse_file import svm_read_problem

def drive_chunking(  train_filename , test_filename,kernel_type_in=None,C_in=None,eps_in=None  ):
	train_y,train_x=svm_read_problem(train_filename)
	# if C_in is None:
	# 	C_in=1
	if eps_in is None:
	 	eps_in=1e-1
	if kernel_type_in is None:
	 	kernel_type_in='linear'
	
	test_y,test_x=svm_read_problem(test_filename)
	init(train_x,train_y,kernel_type_in,C_in,eps_in)
	driver()
	print "Training Accuracy:\n" 
	print get_training_accuracy()

	print "Testing Accuracy:\n"
	print get_test_accuracy(test_x,test_y)


if __name__ == '__main__':
	drive_chunking("../data/leu","../data/leu.t")	
