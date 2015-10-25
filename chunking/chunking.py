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
C=10
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


	if kernel_type_in is None:
		kernel_type='linear'
	else:
		kernel_type=kernel_type_in	
	if C_in is None:
		C=10
	else:	
		C=C_in
	if eps is None:
		eps=1e-5
	else:	
		eps=eps_in
	if max_iter_in is None:
		max_iter=50
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
			if ( f(point[i])>=0 and target[i]==1 ) or (f(point[i])<0 and target[i]==-1  ):
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
	omega=np.zeros(num_dimensions)
	for i in xrange(num_vectors):
		omega=omega+(target[i]*alpha[i])*point[i]
	return omega

def f(z):
	return np.dot(omega,z)+b

def check_stopping_criteria():
	for i in xrange(num_vectors):
		y=target[i]
		alph=alpha[i]
		E=f(point[i]) - y
		r=E*y
		if ((r < -eps and alph<C  ) or ( r>eps and alph>0  )  ) :
			return False
	return True


def compute_gram_matrix(X):
	n,d=X.shape
	res=np.diag(np.ones(n))
	for i in xrange(n):
		for j in xrange(n):
			res[i][j]=res[j][i]=kernel(X[i],X[j])
	return res		

def solve_QP(X,Y):
	n,d= X.shape
	K = compute_gram_matrix(X)

	P = cvxopt.matrix(np.outer(Y, Y) * K)
	q = cvxopt.matrix(-1 * np.ones(n))

	G_std = cvxopt.matrix(np.diag(np.ones(n) * -1))
	h_std = cvxopt.matrix(np.zeros(n))

	G_slack = cvxopt.matrix(np.diag(np.ones(n)))
	h_slack = cvxopt.matrix(np.ones(n) * C)

	G = cvxopt.matrix(np.vstack((G_std, G_slack)))
	h = cvxopt.matrix(np.vstack((h_std, h_slack)))

	A = cvxopt.matrix(Y, (1, n))
	b = cvxopt.matrix(0.0)

    # solvers.options['show_progress'] = qpProgressOut

	solution = solvers.qp(P, q, G, h, A, b)
	print "QP Solution"
	print solution['x']    # solution = solvers.qp(P, q, G_slack, h_slack, A, b)
	return np.ravel(solution['x'])	# returns alpha

def solve_subset(X,Y):
	n,d=X.shape

	print "X dtype"
	print X.dtype

	print "Y dtype"
	print Y.dtype

	alpha_subset=solve_QP(X,Y)
	omega_subset =np.zeros(d)
	b_subset=0.0
	count=0

	print alpha_subset
	for a,x,y in zip(alpha_subset,X,Y):
		omega_subset=omega_subset+(a*y*x)
	for a,x,y in zip(alpha,X,Y):
		if abs(a)>eps:
			b_subset=y-np.dot(omega_subset,x)
			count+=1
	if count>0:		
		b_subset/=count		
	# print alpha
	return alpha_subset,omega_subset,b_subset

def clean_working_set():
	global working_set_indices
	working_set_indices_new=[]
	for i in working_set_indices:
		if abs(alpha[i])>eps:
			working_set_indices_new.append(i)
	rest_indices=list(set(all_indices)-set(working_set_indices))
	
	violators=[]
	for i in rest_indices:
		print "y"
		print target[i]
		y=target[i]
		alph=alpha[i]
		E=f(point[i]) - y
		print "E"
		print E
		r=E*y
		if(abs(alph)<eps and r<0):
			violators.append((abs(E),i))
		elif(abs(alph-C)<eps and r>0):
			violators.append((abs(E),i))
		elif(abs(alph)>eps and abs(alph)<C-eps) and r>eps:
			violators.append((abs(E),i))
	
	violators.sort(key=lambda tup: tup[1],reverse=True)		

	print "violators"
	print violators

	num_picked=0
	if len(working_set_indices)/2==0 :
		num_picked=4
	else:
		num_picked=len(working_set_indices)/2

	for i in xrange(min(num_picked,len(violators))):
		working_set_indices_new.append(violators[i][1])	
	working_set_indices=working_set_indices_new	


def rand_subset(n,k):
	rand_indices= np.random.permutation(n)[0:k]
	rand_indices.sort()
	return rand_indices


def driver():
	global alpha
	global b
	global omega
	iter=0
	b=0.5
	# clean_working_set()
	working_set_indices=rand_subset(num_vectors,20)
	point_subset=point[working_set_indices,]
	target_subset=target[working_set_indices,]
	alpha_subset,omega_subset,b_subset=solve_subset(point_subset,target_subset)
	if b_subset>eps:
		b=b_subset

	alpha[working_set_indices,]=alpha_subset
	omega=omega_subset

	converged=False
	while(  (not converged ) and iter<max_iter):
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


		print "D_BB"
		print D_BB

		print "y_B"
		print y_B

		print "K_BB"
		print K_BB

		print "Q_BN"
		print Q_BN

		print "working_set_indices"
		print working_set_indices

		print "omega"
		print omega.sum()
		# print D_BB



		size_B=y_B.size
		size_N=y_N.size
		G_std = cvxopt.matrix(np.diag(np.ones(size_B) * -1))
		h_std = cvxopt.matrix(np.zeros(size_B))
		G_slack = cvxopt.matrix(np.diag(np.ones(size_B)))
		h_slack = cvxopt.matrix(np.ones(size_B) * C)
		G = cvxopt.matrix(np.vstack((G_std, G_slack)))
		h = cvxopt.matrix(np.vstack((h_std, h_slack)))
		A = cvxopt.matrix(y_B, (1, size_B))
		# B = cvxopt.matrix(-np.dot(alpha_N,y_N))
		B = cvxopt.matrix(0.0)
		solution = cvxopt.solvers.qp(P, q, G, h, A, B)
		alpha_B_new=solution['x']

		print "solution"
		print alpha_B_new

		alpha[working_set_indices,]=alpha_B_new
		compute_omega()
		converged=check_stopping_criteria()
		clean_working_set()
		iter+=1
	print "converged\n"	
