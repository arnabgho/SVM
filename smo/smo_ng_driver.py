import numpy as np
from smo_ng import smo
from parse_file import svm_read_problem

def drive_smo_ng( filename , C = None , Tol=None , max_passes=None , Kernel= None   ):
	y,x=svm_read_problem(filename)
	if C is None:
		C=1
	if Tol is None:
	 	Tol=1e-5
	if max_passes is None:
	 	max_passes=5
	if Kernel is None:
	 	Kernel='linear'
	
	svm_ob=smo(x,y,C,Tol,max_passes,Kernel)
	svm_ob.driver()
	print svm_ob.get_training_accuracy()

