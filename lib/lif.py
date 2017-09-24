#Simulate LIF neurons
import numpy as np
import numpy.random as rand

class ParamsLIF:
	def __init__(self, dt = 0.001, tr = 0.003, mu = 1, reset = 0, xsigma = 1, n = 2, tau = 1,\
		c = 0.99, sigma = 20):

		self.dt = dt         	#Step size
		self.tr = tr         	#Refractory period
		self.mu = mu            #Threshold
		self.reset = reset      #Reset potential
		self.xsigma = xsigma    #Std dev of input x
		self.n = n              #Number of neurons
		self.tau = tau          #Time constant
		self.c = c              #Correlation between noise inputs
		self.sigma = sigma      #Std dev of noise process

class LIF:
	def __init__(self, params, t = 10):
		self.setup(params, t)

	def setup(self, params = None, t = None):
		if params is not None:
			self.params = params
		if t is not None:
			self.t = t

		#Initialize voltage and spike train variables
		self.T = np.ceil(self.t/self.params.dt).astype(int)
		self.Tr = np.ceil(self.params.tr/self.params.dt).astype(int)
		self.times = np.linspace(0,self.t,self.T)
		
		self.x = 3
		self.W = np.array([5, 5])
		self.V = np.array([8, -4])
		
	def simulate(self):
		v = np.zeros((self.params.n,self.T))
		h = np.zeros((self.params.n,self.T))
		vt = np.zeros(self.params.n)
		r = np.zeros(self.params.n)

		#Generate new noise with each sim
		xi = self.params.sigma*rand.randn(self.params.n+1,self.T)/np.sqrt(self.params.tau)
		xi[0,:] = xi[0,:]*np.sqrt(self.params.c)
		xi[1:,:] = xi[1:,:]*np.sqrt(1-self.params.c)

		#Simulate t seconds
		for i in range(self.T):
			dv = -vt/self.params.tau + np.multiply(self.W,(self.x + xi[0,i] + xi[1:,i]))
			vt = vt + self.params.dt*dv
			#Find neurons that spike
			s = vt>self.params.mu
			#Save the voltages and spikes
			h[:,i] = s.astype(int)
			v[:,i] = vt
			#Make spiking neurons refractory
			r[s] = self.Tr
			#Set the refractory neurons to v_reset
			vt[r>0] = self.params.reset
			vt[vt<self.params.reset] = self.params.reset
			#Decrement the refractory counters
			r[r>0] -= 1

		#Cost function per time point
		C = (self.V[0]*h[0,:]+self.V[1]*h[1,:]-self.x**2)**2

		#True causal effect for each unit
		beta1 = self.V[0]**2 + 2*self.V[0]*self.V[1]*np.mean(h[1,:])-2*self.V[0]*self.x**2
		beta2 = self.V[1]**2 + 2*self.V[0]*self.V[1]*np.mean(h[0,:])-2*self.V[1]*self.x**2
		betas = [beta1, beta2]

		return (v, h, C, betas)

class LSM:
	def __init__(self, params, q, t = 10):
		self.setup(params, t, q)

	def setup(self, params = None, t = None, q = None):
		if params is not None:
			self.params = params
		if t is not None:
			self.t = t
		if q is not None:
			self.q = q

		#Initialize voltage and spike train variables
		self.T = np.ceil(self.t/self.params.dt).astype(int)
		self.Tr = np.ceil(self.params.tr/self.params.dt).astype(int)
		self.times = np.linspace(0,self.t,self.T)
		
		#Input x
		self.x = 3

		#Generate random connectivity matrices
		

		self.W = np.array([5, 5])
		self.V = np.array([8, -4])
		
	def simulate(self):
		v = np.zeros((self.params.n,self.T))
		h = np.zeros((self.params.n,self.T))
		vt = np.zeros(self.params.n)
		r = np.zeros(self.params.n)

		#Generate new noise with each sim
		xi = self.params.sigma*rand.randn(self.params.n+1,self.T)/np.sqrt(self.params.tau)
		xi[0,:] = xi[0,:]*np.sqrt(self.params.c)
		xi[1:,:] = xi[1:,:]*np.sqrt(1-self.params.c)

		#Simulate t seconds
		for i in range(self.T):
			dv = -vt/self.params.tau + np.multiply(self.W,(self.x + xi[0,i] + xi[1:,i]))
			vt = vt + self.params.dt*dv
			#Find neurons that spike
			s = vt>self.params.mu
			#Save the voltages and spikes
			h[:,i] = s.astype(int)
			v[:,i] = vt
			#Make spiking neurons refractory
			r[s] = self.Tr
			#Set the refractory neurons to v_reset
			vt[r>0] = self.params.reset
			vt[vt<self.params.reset] = self.params.reset
			#Decrement the refractory counters
			r[r>0] -= 1

		#Cost function per time point
		C = (self.V[0]*h[0,:]+self.V[1]*h[1,:]-self.x**2)**2

		#True causal effect for each unit
		beta1 = self.V[0]**2 + 2*self.V[0]*self.V[1]*np.mean(h[1,:])-2*self.V[0]*self.x**2
		beta2 = self.V[1]**2 + 2*self.V[0]*self.V[1]*np.mean(h[0,:])-2*self.V[1]*self.x**2
		betas = [beta1, beta2]

		return (v, h, C, betas)
