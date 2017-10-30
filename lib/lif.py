#Simulate LIF neurons
import numpy as np
import numpy.random as rand

class ParamsLIF(object):
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

class ParamsLSM(ParamsLIF):

	def __init__(self, t = 1, q = 100, p = 1, tau_s = .04, I0 = 20, spectral_radius = 0.95, **kwargs):
		self.t = t
		self.q = q
		self.p = p
		self.tau_s = tau_s
		self.I0 = I0
		self.spectral_radius = spectral_radius
		super(ParamsLSM, self).__init__(**kwargs)

class LIF(object):
	def __init__(self, params, t = 10):
		self.setup(params, t)

	def setup(self, params = None, t = None):
		#print("I am setting up LIF")
		if params is not None:
			self.params = params
		if t is not None:
			self.t = t

		#Initialize voltage and spike train variables
		self.T = np.ceil(self.t/self.params.dt).astype(int)
		self.Tr = np.ceil(self.params.tr/self.params.dt).astype(int)
		self.times = np.linspace(0,self.t,self.T)
		
		self.x = 0
		self.W = np.array([5, 5])
		self.V = np.array([8, -4])

		self.keepstate = True
		self.vt = np.zeros(self.params.n)

		
	def simulate(self):
		v = np.zeros((self.params.n,self.T))
		h = np.zeros((self.params.n,self.T))

		if not self.keepstate:
			self.vt = np.zeros(self.params.n)

		vt = self.vt

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

		self.vt = vt

		return (v, h, C, betas)

class LSM(LIF):
	"""A class for a liquid state machine

	Something that simulates a randomly connected network of LIF units and an input x
	"""

	def setup(self, params = None, t = None):
		#print("I am setting up LSM")
		#print(params)
		if t is not None:
			self.t = t
		if params is not None:
			self.params = params

		mu_w = 50
		sigma_w = 30

		mu_u = 5
		sigma_u = 2

		#Fix the random seed
		#rand.seed(42)

		q = self.params.q
		p = self.params.p 

		#Initialize voltage and spike train variables
		self.T = np.ceil(self.params.t/self.params.dt).astype(int)
		self.Tr = np.ceil(self.params.tr/self.params.dt).astype(int)
		self.times = np.linspace(0,self.params.t,self.T)

		#Generate sparse, random connectivity matrices
		self.W = np.zeros((q,q))
		self.U = np.zeros((q,p))

		#Sparse connectivity, only 10% neurons are connected to one another
		#q = 10

		m = np.ceil(0.1*q).astype(int);
		#80% excitatory, 20% inhibitory
		q_e = np.ceil(0.8*q).astype(int);
		q_i = q - q_e;

		#Choose excitatory neurons
		ex = rand.choice(q, q_e, replace=False)
		#W = np.zeros((q,q))

		for idx in range(q):
			conn = rand.choice(q, m, replace=False)
			weights = np.maximum(rand.randn(m)*sigma_w + mu_w, 0)
			if idx not in ex:
				weights *= -1
			self.W[conn,idx] = weights

		#Scale by spectral radius...
		#Compute the spectral radius of the weights and rescale
		radius = np.max(np.abs(np.linalg.eigvals(self.W)))
		self.W = self.W * (self.params.spectral_radius / radius)

		ex = rand.choice(q, np.ceil(0.5*q).astype(int), replace=False)
		for idx in range(q):
			weights = np.maximum(rand.randn(p)*sigma_u + mu_u, 0)
			if idx not in ex:
				weights *= -1
			self.U[idx,:] = weights


		self.I0 = np.zeros((q,1))
		self.I0[:,0] = np.maximum(rand.randn(q)*sigma_u+mu_u, 0)
		self.st = np.zeros((self.params.q,1))

	def simulate(self, x):
		s = np.zeros((self.params.q,self.T))
		vt = np.zeros((self.params.q,1))
		rt = np.zeros((self.params.q,1))
		sp = np.zeros((self.params.q,1))
		st = self.st

		#Simulate t seconds
		for i in range(self.T):
			#Compute dynamics
			ds = -st/self.params.tau_s + sp/self.params.tau_s
			st = st + self.params.dt*ds
			dv = -vt/self.params.tau + np.dot(self.U, x)[:,None] + np.dot(self.W, st) + self.I0
			vt = vt + self.params.dt*dv
			#print dv.shape
			#print np.dot(self.U, x).shape
			#Find neurons that spike
			sp = vt>self.params.mu
			#Make spiking neurons refractory
			rt[sp] = self.Tr
			#Set the refractory neurons to v_reset
			vt[rt>0] = self.params.reset
			vt[vt<self.params.reset] = self.params.reset
			#Decrement the refractory counters
			rt[rt>0] -= 1
			#Save the output
			s[:,i] = st[:,0]

		self.st = st

		return s

class LSM_const(LSM):
	def simulate(self, x):
		s = np.ones((self.params.q,self.T))
		return s

class LIF_3layer(object):
	def __init__(self, params, t = 1, nx = 2, no = 1, tau_s = 0.020, onrate = 200, offrate = 100, alpha = 30, spectral_radius = 0.95):
		self.tau_s = tau_s
		self.onrate = onrate
		self.offrate = offrate
		self.alpha = alpha
		self.spectral_radius = spectral_radius
		self.setup(params, t, nx, no)

	def setup(self, params = None, t = None, nx = None, no = None):
		#print("I am setting up LIF")
		if params is not None:
			self.params = params
		if t is not None:
			self.t = t
		if nx is not None:
			self.nx = nx
		if no is not None:
			self.no = no

		#Initialize voltage and spike train variables
		self.T = np.ceil(self.t/self.params.dt).astype(int)
		self.Tr = np.ceil(self.params.tr/self.params.dt).astype(int)
		self.times = np.linspace(0,self.t,self.T)
		
		self.x = np.ones((self.nx,1))
		self.W = self.alpha*np.random.rand(self.params.n, self.nx)-self.alpha/3.
		self.U = self.alpha*np.random.rand(self.no, self.params.n)-self.alpha/3.

		t_filter = np.linspace(0, 1, 2000)
		exp_filter = np.exp(-t_filter/self.tau_s)
		self.exp_filter = exp_filter/np.sum(exp_filter)
		self.ds = exp_filter[0]

		self.vth = np.zeros(self.params.n)
		self.vto = np.zeros(self.no)

	def simulate(self, x):
		vh = np.zeros((self.params.n,self.T))
		hh = np.zeros((self.params.n,self.T))
		vo = np.zeros((self.no, self.T))
		ho = np.zeros((self.no, self.T))
		sx = np.zeros((self.nx, self.T))
		so = np.zeros((self.params.n, self.T))

		#Generate new noise with each sim
		xi = self.params.sigma*rand.randn(self.nx+1,self.T)/np.sqrt(self.params.tau)
		#Common noise
		xi[0,:] = xi[0,:]*np.sqrt(self.params.c)
		#Indep noise
		xi[1:,:] = xi[1:,:]*np.sqrt(1-self.params.c)
		#Output noise
		xo = self.params.sigma*rand.randn(self.no,self.T)/np.sqrt(self.params.tau)

		#Generate a noisy input process: filtered Poisson spiking
		for i in range(self.nx):
			rate = self.onrate if x[i] == 1 else self.offrate
			#print("x%d = %f"%(i, rate))
			ip = np.random.rand(self.T) < rate*self.params.dt
			sx[i,:] = np.convolve(ip, self.exp_filter)[0:self.T]

		vt = self.vth
		dv = np.zeros(self.params.n)
		r = np.zeros(self.params.n)
		#Simulate t seconds for the hidden layer
		for t in range(self.T):
			dv = -vt/self.params.tau + np.squeeze(np.dot(self.W, sx[:,t]+xi[0,t] + xi[1:,t]))
			vt = vt + self.params.dt*dv
			#Find neurons that spike
			s = vt>self.params.mu
			#Make spiking neurons refractory
			r[s] = self.Tr
			#Save the voltages and spikes
			hh[:,t] = s.astype(int)
			vh[:,t] = vt
			#Set the refractory neurons to v_reset
			vt[r>0] = self.params.reset
			vt[vt<self.params.reset] = self.params.reset
			#Decrement the refractory counters
			r[r>0] -= 1

		self.vth = vt

		vt = self.vto

		#Simulate t seconds output neuron
		for i in range(self.params.n):
			so[i,:] = np.convolve(hh[i,:], self.exp_filter)[0:self.T]

		vt = np.zeros(self.no)
		dv = np.zeros(self.no)
		r = np.zeros(self.no)
		for t in range(self.T):
			#Determine the output neuron's output
			dv = -vt/self.params.tau + np.squeeze(np.dot(self.U, so[:,t]+xo[:,t]))
			vt = vt + self.params.dt*dv
			#Find neurons that spike
			s = vt>self.params.mu
			#Make spiking neurons refractory
			r[s] = self.Tr
			ho[:,t] = s.astype(int)
			vo[:,t] = vt
			#Set the refractory neurons to v_reset
			vt[r>0] = self.params.reset
			vt[vt<self.params.reset] = self.params.reset
			#Decrement the refractory counters
			r[r>0] -= 1

		self.vto = vt

		return vh, hh, vo, ho, so, sx