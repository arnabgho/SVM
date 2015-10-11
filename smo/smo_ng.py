import numpy as np
from random import randint
class smo:
	
	# self.num_vectors=0
	# self.alpha_i_old=0
	# self.alpha_j_old=0
	#kernel_type="linear"
	def __init__(self,X,Y,C,Tol,max_passes=None,Kernel=None):
		self.x=X
		self.y=Y
		self.C=C
		self.tol=Tol
		self.num_vectors=len(X)
		self.alpha_j_old=0
		self.alpha_i_old=0
		self.max_passes=1
		self.b=0
		if Kernel is None:
			self.kernel_type="linear"
		else:
			self.kernel_type=Kernel
		if max_passes is None:
			self.max_passes=10
		else:
			self.max_passes=max_passes

	def compute_kernel(self,x,y):
		if self.kernel_type=="linear":
			return np.dot(x,y)


	def get_L(self,i,j):
		if self.y[i]!=self.y[j]:
			return max(0,self.alpha_j_old-self.alpha_i_old)
		else:
			return max(0,self.alpha_i_old+self.alpha_j_old-self.C)
	
	def get_H(self,i,j):
		if self.y[i]!=self.y[j]:
			return min(self.C,self.C+self.alpha_j_old-self.alpha_i_old)
		else:
			return min(self.C,self.alpha_i_old+self.alpha_j_old)
		
	def get_eta(self,i,j):
		return 2*self.compute_kernel(self.x[i],self.x[j])-self.compute_kernel(self.x[i],self.x[i])-self.compute_kernel(self.x[j],self.x[j])
		

	def get_self_alphaj(self,i,j,H,L,Ei,Ej,eta):
		new_alpha_j=self.alpha_j_old-self.y[j]*(Ei-Ej)/eta
		if new_alpha_j>H:
			return H
		elif new_alpha_j<L:
			return L
		else:
			return new_alpha_j

	def get_b1(self,i,j,Ei,Ej,b):
		return self.b-Ei-self.y[i]*( self.alpha[i]-self.alpha_i_old )*self.compute_kernel(self.x[i],self.x[i])-self.y[j]*(self.alpha[j]-self.alpha_j_old)*self.compute_kernel(self.x[i],self.x[j])

	def get_b2(self,i,j,Ei,Ej,b):
		return self.b-Ej-self.y[i]*( self.alpha[i]-self.alpha_i_old )*self.compute_kernel(self.x[i],self.x[j])-self.y[j]*(self.alpha[j]-self.alpha_j_old)*self.compute_kernel(self.x[j],self.x[j])

	def get_b( self , b1 , b2  , i, j ) :
		if	(0<self.alpha[i] and self.alpha[i]<self.C) :
			return b1
		elif (0<self.alpha[j] and self.alpha[j]<self.C) :
			return b2
		else:
			return 0.5*(b1+b2)

	def f(self,z):	
		ans=self.b
		for i in xrange(self.num_vectors):
			ans+=self.alpha[i]*self.compute_kernel(self.x[i],z)
		return ans

	def get_training_accuracy(self):
		correct=0.0
		for i in xrange(self.num_vectors):
			if self.f(self.x[i])*self.y[i]>=0:
				correct+=1.0
		return 	correct/self.num_vectors

	def driver(self):
		self.alpha=np.zeros( self.num_vectors , float  )
		passes=0
		while passes<self.max_passes:
			num_changed_self_alphas=0
			for i in xrange( self.num_vectors  ):
				Ei=self.f(self.x[i])-self.y[i]
				if (self.y[i]*Ei<-self.tol and self.alpha[i]<self.C  ) or ( self.y[i]*Ei > self.tol and self.alpha[i]>0  )   :
					j=randint(0,self.num_vectors-1)
					while(j==i):
						j=randint(0,self.num_vectors-1)
					Ej=self.f(self.x[j])-self.y[j]
					self.alpha_i_old=self.alpha[i]
					self.alpha_j_old=self.alpha[j]	 
					L= self.get_L(i,j) # Compute L 
					H= self.get_H(i,j) # Compute H
					if(L==H):
						continue
					# Compute eta
					eta=self.get_eta(i,j)	
					if(eta>=0):
						continue
					# Compute new value for self.alpha[j]
					self.alpha[j]=self.get_self_alphaj(i,j,H,L,Ei,Ej,eta)	
					if(abs(self.alpha[j]-self.alpha_j_old)<1e-5):
						continue
					self.alpha[i]=self.alpha_i_old+self.y[i]*self.y[j]*(self.alpha_j_old-self.alpha[j])	
					# Compute Value for self.alpha[i]
					# Compute b1 & b2
					b1=self.get_b1(i,j,Ei,Ej,self.b)
					b2=self.get_b2(i,j,Ej,Ej,self.b)
					# Compute b
					self.b=self.get_b(b1,b2,i,j)
					num_changed_self_alphas+=1
			if(num_changed_self_alphas==0):
				passes+=1
			else:
			 	passes=0



				
