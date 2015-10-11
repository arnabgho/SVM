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

	def takeStep( i1,i2 ):
		if (i1==i2) : 
			 return 0
		alph1=   #  Lagrangian multiplier for i1
		y1=target[i1]
		E1=   	 # SVM output on point[i1]-y1
		s=y1*y2
		# Compute L , H
		if (L==H):
			return 0
		
		k11=kernel( point[i1] , point[i1]  )
		k12= kernel( point[i1] , point[i2]  )
		k22= kernel( point[i2] , point[i2]  )
		eta=2*k12-k11-k22
		if eta<0:
			a2=alph2-y2*(E1-E2)/eta
			if a2<L:
				a2=L
			else if a2>H :
				a2=H
		
		else:
			Lobj=		#objective function at a2=L
			Hobj=		#objective function at a2=H
			if Lobj>Hobj+eps:
				a2=L
			else if Lobj<Hobj-eps :
				a2=H
			else
				a2=alph2
				
		if a2<1e-8:
			a2=0
		else if a2>C-1e-8:
			a2=C
		if abs(a2-alph2)<eps*(a2+alph2+eps):
			return 0
		a1=alph1+s*(alph2-a2)
		# Update threshold to reflect change in Lagrange Multipliers
		# Update weight vector to reflect change in a1 & a2 , if Linear SVM
		# Update Error Cache using new Lagrange Multipliers
		# Store a1 in the alpha array
		# Store a2 in the alpha array
		return 1 

		 

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
				 
		
