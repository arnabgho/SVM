import numpy as np
from smo_sparse import *
from parse_file import svm_read_problem_sparse

def drive_smo_sparse(  train_filename , test_filename,kernel_type_in=None,C_in=None,eps_in=None  ):
	train_y,train_x=svm_read_problem_sparse(train_filename)
	if C_in is None:
		C_in=1
	if eps_in is None:
	 	eps_in=1e-5
	if kernel_type_in is None:
	 	kernel_type_in='linear'
	
	test_y,test_x=svm_read_problem_sparse(test_filename)
	init(train_x,train_y,kernel_type_in,C_in,eps_in)
	driver()
	print "Training Accuracy:\n" 
	print get_training_accuracy()

	print "Testing Accuracy:\n"
	print get_test_accuracy(test_x,test_y)