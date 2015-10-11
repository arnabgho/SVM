import numpy as np
from random import randint
#import cvxopt



num_vectors=0
num_dimensions=0
point=np.zeros(1,float)
target=np.zeros(1,float)
alpha=np.zeros(1,float)
target=np.zeros(1,float)
threshold=np.zeros(1,float)
C=0
eps=0
kernel_type='linear'
b=0
omega=np.zeros(1,float)
num_zero=0
num_C=0

def kernel(x,y):
	if kernel_type=='linear':
		return np.dot(x,y)

def get_training_accuracy():
		correct=0.0
		for i in xrange(num_vectors):
			if ( f(point[i])>=0 and target[i]==1 ) or (f(point[i])<0 and target[i]==-1  )>=0:
				correct+=1.0
		return 	correct/num_vectors

def get_test_accuracy(test_x,test_y):
	num_vectors_test=len(test_x)
	correct=0.0
	for i in xrange(num_vectors_test):
		if ( f(test_x[i])>=0.0 and test_y[i]==1 ) or (f(test_x[i])<0 and test_y[i]==-1  ) :
			correct+=1.0
	return correct/num_vectors_test		

def init(data_in,target_in,kernel_type_in=None,C_in=None,eps_in=None):
	global num_vectors
	global num_dimensions
	global point
	global target
	global alpha
	global target
	global threshold
	global C
	global eps
	global kernel_type
	global b
	global omega
	num_vectors=data_in.shape[0]
	num_dimensions=data_in.shape[1]
	omega=np.zeros(num_dimensions,float)
	point=data_in
	target=target_in
	alpha=np.zeros(num_vectors,float)
	threshold=np.zeros(num_vectors,float)
	num_zero=num_vectors
	num_C=0

	if kernel_type_in is None:
		kernel_type='linear'
	else:
		kernel_type=kernel_type_in	
	if C_in is None:
		C=1
	else:	
		C=C_in
	if eps is None:
		eps=1e-3
	else:	
		eps=eps_in

def f(z):	
	ans=-b
	# for i in xrange(num_vectors):
		# ans+=alpha[i]*kernel(point[i],z)
	ans+=np.dot(omega,z)	
	return ans

def get_b1(i,j,Ei,Ej,b,a1,a2,alph1,alph2):
	return b+Ei+target[i]*( a1-alph1 )*kernel(point[i],point[i])+target[j]*(a2-alph2)*kernel(point[i],point[j])

def get_b2(i,j,Ei,Ej,b,a1,a2,alph1,alph2):
	return b+Ej+target[i]*( a1-alph1 )*kernel(point[i],point[j])+target[j]*(a2-alph2)*kernel(point[j],point[j])


def get_b( b1 , b2  , i, j ) :
	if	(0<alpha[i] and alpha[i]<C) :
		return b1
	elif (0<alpha[j] and alpha[j]<C) :
		return b2
	else:
		return 0.5*(b1+b2)

def takeStep( i1,i2 ):
	global num_vectors
	global num_dimensions
	global point
	global target
	global alpha
	global target
	global threshold
	global C
	global eps
	global kernel_type
	global omega
	global num_C
	global num_zero
	global b
	if (i1==i2) : 
		 return 0
	alph1=  alpha[i1]  #  Lagrangian multiplier for i1
	alph2=  alpha[i2]
	y1=target[i1]
	y2=target[i2]
	E1= f(point[i1])-y1  	 # SVM output on point[i1]-y1
	E2= f(point[i2])-y2
	s=y1*y2
	# Compute L , H

	if s<0:
		L=max(0,alph2-alph1)
		H=min(C,C+alph2-alph1)
	else:
		L=max(0,alph2+alph1-C)
		H=min(C,alph2+alph1)
			

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
		elif a2>H :
			a2=H
	
	else:
		a1_L=alph1+s*(alph2-L)
		a1_H=alph1+s*(alph2-H)
		omega_L=omega+y1*(a1_L-alph1)*point[i1]+y2*(L-alph2)*point[i2]
		omega_H=omega+y1*(a1_L-alph1)*point[i1]+y2*(H-alph2)*point[i2]
		alpha_L=alpha
		alpha_L[i1]=a1_L
		alpha_L[i2]=L
		alpha_H=alpha
		alpha_H[i1]=a1_H
		alpha_H[i2]=H		

		Lobj=	compute_objective_function(omega_L,alpha_L)	#objective function at a2=L
		Hobj=	compute_objective_function(omega_H,alpha_H)	#objective function at a2=H
		if Lobj>Hobj+eps:
			a2=L
		elif Lobj<Hobj-eps :
			a2=H
		else:
			a2=alph2
			
	if a2<1e-8:
		a2=0
	elif a2>C-1e-8:
		a2=C
	if abs(a2-alph2)<eps*(a2+alph2+eps):
		return 0
	a1=alph1+s*(alph2-a2)
	# Update threshold to reflect change in Lagrange Multipliers
	# Update weight vector to reflect change in a1 & a2 , if Linear SVM
	# Update Error Cache using new Lagrange Multipliers
	b1=get_b1(i1,i2,E1,E2,b,a1,a2,alph1,alph2)
	b2=get_b2(i1,i2,E1,E2,b,a1,a2,alph1,alph2)
	b=get_b( b1 , b2  , i1, i2 )
	omega=omega+y1*(a1-alph1)*point[i1]+y2*(a2-alph2)*point[i2]
	alpha[i1]=a1
	alpha[i2]=a2

	if abs(alph1-C)<eps:
		num_C-=1
	if abs(alph2-C)<eps:
		num_C-=1
	if abs(alph1)<eps:
		num_zero-=1
	if abs(alph2)<eps:
		num_zero-=1		

	if abs(a1-C)<eps:
		num_C+=1
	if abs(a2-C)<eps:
		num_C+=1
	if abs(a1)<eps:
		num_zero+=1
	if abs(a2)<eps:
		num_zero+=1			
	# Store a1 in the alpha array
	# Store a2 in the alpha array
	return 1 

def compute_objective_function( omega , alpha ):
	obj=-0.5*np.dot(omega,omega)
	for a in alpha:
		obj+=a
	return obj	
		 

def examineExample( i2  ):
	global num_vectors
	global num_dimensions
	global point
	global target
	global alpha
	global target
	global threshold
	global C
	global eps
	global kernel_type
	global omega
	global num_zero
	global num_C
	y2=target[i2]
	alph2=alpha[i2]
	E2=f(point[i2]) - y2
	r2=E2*y2
	if ((r2 < -eps and alph2<C  ) or ( r2>eps and alph2>0  )  ) :
		if ( num_vectors - num_zero - num_C >1  ) :
			i1=second_choice_heuristic(i2,E2)
			if takeStep(i1,i2):
				return 1
		rand_point=randint(0,num_vectors-1)
		for x in xrange(num_vectors):
			i1=(x+rand_point)%num_vectors
			if (alpha[i1]!=0 and alpha[i1]!=C):
				if takeStep(i1,i2):
					return 1
					
		
		for x in xrange(num_vectors):
			i1=(x+rand_point)%num_vectors
			if takeStep (i1,i2):
				return 1
	
	return 0
		

def second_choice_heuristic(i2,E2):
	global num_vectors
	global num_dimensions
	global point
	global target
	global alpha
	global target
	global threshold
	global C
	global eps
	global kernel_type
	global omega_H

	best=-1
	if E2>0:
		minim=1e9
		for i1 in xrange(num_vectors):
			if i1==i2:
				continue
			else :
				E=f(point[i1])-target[i1]
				if E<minim:
					minim=E
					best=i1				

	else:
		maxim=-1e9
		for i1 in xrange(num_vectors):
			if i1==i2:
				continue
			else :
				E=f(point[i1])-target[i1]
				if E>maxim:
					maxim=E
					best=i1			
	
	return best


def driver():
	global num_vectors
	global num_dimensions
	global point
	global target
	global alpha
	global target
	global threshold
	global C
	global eps
	global kernel_type
	global omega
	numChanged=0
	examineAll=1
	while (numChanged>0 or examineAll):
		numChanged=0
		if examineAll:
			for i in xrange(num_vectors):
				numChanged+=examineExample(i)
		else:
			for i in xrange(num_vectors):
				if alpha[i]!=0 and alpha[i]!=C:
					numChanged+=examineExample(i)
		if examineAll==1:
			examineAll=0
		elif numChanged==0:
			examineAll=1
				 
		
