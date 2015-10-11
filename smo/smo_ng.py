import numpy as np
import random as randint
class smo:
	
	num_vectors=0
	alpha_i_old=0
	alpha_j_old=0
	kernel_type="linear"
	def __init__(self,X,Y,Kernel,C,Tol):
		self.x=X
		self.y=Y
		self.C=C
		self.tol=Tol
		self.kernel_type=Kernel

	def compute_kernel(self,x,y):
		if kernel_type=="linear":
			return np.dot(x,y)


	def get_L(self,i,j):
		if y[i]!=y[j]:
			return max(0,alpha_j_old-alpha_i_old)
		else:
			return max(0,alpha_i_old+alpha_j_old-C)
	
	def get_H(self,i,j):
		if y[i]!=y[j]:
			return min(C,C+alpha_j_old-alpha_i_old)
		else:
			return min(C,alpha_i_old+alpha_j_old)
		
	def get_eta(self,i,j):
		return 2*compute_kernel(x[i],x[j])-compute_kernel(x[i],x[i])-compute_kernel(x[j],x[j])
		

	def get_alphaj(self,i,j,H,L,Ei,Ej,eta):
		new_alpha_j=alpha_j_old-y[j]*(Ei-Ej)/eta
		if new_alpha_j>H:
			return H
		elif new_alpha_j<L:
			return L
		else:
			return new_alpha_j

	def get_b1(self,i,j,Ei,b):
		return b-Ei-y[i]*( alpha[i]-alpha_i_old )*compute_kernel(x[i],x[j])-y[j]*(alpha[j]-alpha_j_old)*compute_kernel(x[i],x[j])

	def get_b2(self,i,j,Ei,b):
		return b-Ej-y[i]*( alpha[i]-alpha_i_old )*compute_kernel(x[i],x[j])-y[j]*(alpha[j]-alpha_j_old)*compute_kernel(x[i],x[j])

	def get_b( self , b1 , b2  ) :
		if	(0<alpha[i] and alpha[i]<C) :
			return b1
		elif (0<alpha[j] and alpha[j]<C) :
			return b2
		else:
			return 0.5*(b1+b2)

	def f(self,z):	
		ans=b
		for i in xrange(num_vectors):
			ans+=alpha[i]*compute_kernel(x[i],z)
		return ans	

	def driver(self):
		alpha=np.zeros( num_vectors , float  )
		passes=0
		while passes<max_passes:
			for i in xrange( num_vectors  ):
				Ei=f(x[i])-y[i]
				if (y[i]*Ei<-tol and alpha[i]<C  ) or ( y[i]*Ei > tol and alpha[i]>0  )   :
					j=randint(0,num_vectors-1)
					while(j==i):
						j=randint(0,num_vectors-1)
					Ej=f(x[j])-y[j]
					alpha_i_old=alpha[i]
					alpha_j_old=alpha[j]	 
					L= get_L(i,j) # Compute L 
					H= get_H(i,j) # Compute H
					if(L==H):
						continue
					# Compute eta
					eta=get_eta(i,j)	
					if(eta>=0):
						continue
					# Compute new value for alpha[j]
					alpha[j]=get_alphaj(i,j,H,L,Ei,Ej,eta)	
					if(abs(alpha[j]-alpha_j_old)<1e-5):
						continue
					alpha[i]=alpha_i_old+y[i]*y[j](alpha_j_old-alpha[j])	
					# Compute Value for alpha[i]
					# Compute b1 & b2
					b1=get_b1(i,j,Ei,b)
					b2=get_b2(i,j,Ej,b)
					# Compute b
					b=get_b(b1,b2)
					num_changed_alphas+=1
			if(num_changed_alphas==0):
				passes+=1
			else:
			 	passes=0



				
