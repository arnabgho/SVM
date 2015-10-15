import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import numpy as np
import cvxopt
from cvxopt import matrix
from cvxopt import solvers

K=np.diag([1.0,1.0])
D=np.diag([1.0,1.0])
num_vectors=0
num_dimensions=0
point=np.zeros(1,float)
point_matrix=matrix(point)
target=np.zeros(1,float)
alpha=np.zeros(1,float)
target=np.zeros(1,float)
C=0
eps=0
kernel_type='linear'
b=0
omega=np.zeros(1,float)
num_zero=0
num_C=0
working_set_indices=[]
max_iter=2
all_indices=[]

def init(data_in,target_in,kernel_type_in=None,C_in=None,eps_in=None,max_iter_in=None):
	global num_vectors
	global num_dimensions
	global point
	global target
	global alpha
	global target
	global C
	global eps
	global kernel_type
	global b
	global omega
	global working_set_indices
	global point_matrix
	global all_indices
	global max_iter

	num_vectors=data_in.shape[0]
	num_dimensions=data_in.shape[1]
	omega=np.zeros(num_dimensions,float)
	point=data_in
	target=target_in
	alpha=np.zeros(num_vectors,float)
	point_matrix=matrix(point)
	all_indices=range(num_vectors)

	print "all_indices"
	print all_indices
	if kernel_type_in is None:
		kernel_type='linear'
	else:
		kernel_type=kernel_type_in	
	if C_in is None:
		C=1
	else:	
		C=C_in
	if eps is None:
		eps=1e-5
	else:	
		eps=eps_in
	if max_iter_in is None:
		max_iter=1000
	else:
		max_iter=max_iter_in	
	compute_kernel_matrix()	

def kernel(x,y):
	if kernel_type=='linear':
		return np.dot(x,y)

def compute_kernel_matrix():
	global K
	global D
	K=D=np.diag(np.ones(num_vectors))
	for i in xrange(num_vectors):
		for j in xrange(i,num_vectors):
			K[i][j]=K[i][j]=kernel(point[i],point[j])
			D[i][j]=D[j][i]=K[i][j]*target[i]*target[j]


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

def compute_omega():
	global omega
	omega= np.array(matrix(alpha).T*point_matrix)
	return omega

def f(z):
	return np.dot(omega,z)

def check_stopping_criteria():
	for i in xrange(num_vectors):
		y=target[i]
		alph=alpha[i]
		E=f(point[i]) - y
		r=E*y
		if ((r < -eps and alph<C  ) or ( r>eps and alph>0  )  ) :
			return False
	return True

def clean_working_set():
	global working_set_indices
	working_set_indices_new=[]
	for i in working_set_indices:
		if abs(alpha[i])>eps:
			working_set_indices_new.append(i)
	rest_indices=list(set(all_indices)-set(working_set_indices))
	
	violators=[]
	for i in rest_indices:
		y=target[i]
		alph=alpha[i]
		E=f(point[i]) - y
		r=E*y
		if(abs(alph)<eps and E<0):
			violators.append((abs(E),i))
		elif(abs(alph-C)<eps and E>0):
			violators.append((abs(E),i))
		elif(abs(alph)>eps and abs(alph)<C-eps) and E>eps:
			violators.append((abs(E),i))
	
	violators.sort(key=lambda tup: tup[1],reverse=True)		

	num_picked=0
	if len(working_set_indices)/2==0 :
		num_picked=10
	else:
		num_picked=len(working_set_indices)/2

	for i in xrange(min(num_picked,len(violators))):
		working_set_indices_new.append(violators[i][1])	
	working_set_indices=working_set_indices_new	


def driver():
	global alpha
	iter=0
	converged=False
	while(  (not converged ) and iter<max_iter):
		clean_working_set()
		rest_indices=list(set(all_indices)-set(working_set_indices))
		y_B=target[ working_set_indices , ]
		alpha_B=alpha[working_set_indices,]
		y_N=target[rest_indices,]
		alpha_N=alpha[rest_indices,]
		K_B=K[working_set_indices,]
		K_BB=K_B[:,working_set_indices]
		K_BN=K_B[:,rest_indices]

		D_B=D[working_set_indices,]
		D_BB=D_B[:,working_set_indices]
		D_BN=D_B[:,rest_indices]
		
		alpha_y_N=alpha_N*y_N
		Q_BN=np.array(matrix(alpha_y_N).T*matrix(K_BN).T)
		Q_BN=Q_BN*y_B
		ones=np.ones(Q_BN.size)
		P=cvxopt.matrix(D_BB)
		q=cvxopt.matrix((Q_BN-ones).T)


		print "working_set_indices"
		print working_set_indices

		# print D_BB



		# size_B=y_B.size
		# size_N=y_N.size
		# G_std = cvxopt.matrix(np.diag(np.ones(size_B) * -1))
		# h_std = cvxopt.matrix(np.zeros(size_B))
		# G_slack = cvxopt.matrix(np.diag(np.ones(size_B)))
		# h_slack = cvxopt.matrix(np.ones(size_B) * C)
		# G = cvxopt.matrix(np.vstack((G_std, G_slack)))
		# h = cvxopt.matrix(np.vstack((h_std, h_slack)))
		# A = cvxopt.matrix(y_B, (1, size_B))
		# b = cvxopt.matrix(-np.dot(alpha_N,y_N))
		# solution = cvxopt.solvers.qp(P, q, G, h, A, b)
		# alpha_B_new=solution['x']
		# alpha[working_set_indices,]=alpha_B_new
		# compute_omega()
		# converged=check_stopping_criteria()
		# iter+=1
	print "converged\n"	
