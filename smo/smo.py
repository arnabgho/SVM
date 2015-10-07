import numpy as np
from random import randint
#import cvxopt

class smo:

	num_vectors=0
	num_dimensions=0
#	point
#	target
#	alpha
	

	def __init__(self,data,target,kernel="dot",C,eps):
		self.num_vectors=data.shape[0]
		self.num_dimensions=data.shape[1]
		self.point=data
		self.target=target
		self.alpha=np.zeros(num_vectors,float)
		self.threshold=np.zeros(num_vectors,float)
		self.C=C
		self.tol=eps

	def examineExample( self,i2  ):
		y2=target[i2]
		alph2=alpha[i2]
		E2=classify(i2) - y2
		r2=E2*y2
		if ((r2 < -tol && alph2<C  ) || ( r2>tol && alph2>0  )  ) :
			if ( num_vectors - num_zero - num_C >1  ) :
				i1=second_choice_heuristic()
				if takeStep(i1,i2):
					return 1
			rand_point=randint(0,num_vectors-1)
			for x in xrange(num_vectors):
				i1=(x+rand_point)%num_vectors
				if (alpha[i1]!=0 && alpha[i1]!=C):
					if takeStep(i1,i2):
						return 1
						
			
			for x in xrange(num_vectors):
				i1=(x+rand_point)%num_vectors
				if takeStep (i1,i2):
					return 1
		
		return 0
		



	def driver(self):
		numChanged=0
		examineAll=1
		while (numChanged>0 | examineAll):
			numChanged=0
			if examineAll:
				for i in xrange(num_vectors):
					numChanged+=examineExample(point[i])
			else:
				for i in xrange(num_vectors):
					if alpha[i]!=0 && alpha[i]!=C:
						numChanged+=examineExample(point[i])
			if examineAll==1:
				examineAll=0
			else if numChanged==0:
				examineAll=1
				 
		
