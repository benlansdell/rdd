#Simulate LIF neurons
import numpy as np
import numpy.random as rand
import scipy

def convolve_online(s, h, kernel, t_offset):
    if len(s.shape) > 1:
        n = s.shape[0]
        for i in range(n):
            for idx in np.nonzero(h[i,:])[0]:
                st = t_offset + idx
                en = min(s.shape[0], st + kernel.shape[0])
                ln = en-st
                #print st,en,ln,idx,t_offset
                s[i,st:en] += kernel[0:ln]
    else:
        for idx in np.nonzero(h)[0]:
            st = t_offset + idx
            en = min(s.shape[0], st + kernel.shape[0])
            ln = en-st
            #print st,en,ln,idx,t_offset
            s[st:en] += kernel[0:ln]

def convolve_online_v2(s, sp_idx, time_idx, kernel, t_offset):
    st = t_offset + time_idx
    en = min(s.shape[1], st + kernel.shape[0])
    ln = en-st
    s[sp_idx, st:en] += kernel[0:ln]

def firingrate_LIF(params, W, S):
    #Computes the firing rate as a function of weights W and mean inputs S
    tau = params.tau
    Vth = params.mu
    Vr = params.reset
    Vrest = params.reset
    dt = params.dt
    sigma_xi = params.sigma*dt
    I = np.dot(W, S)
    sigma = sigma_xi*np.sqrt(np.sum(W**2))
    Yth = (Vth - Vrest - I)/sigma
    Yr = (Vr - Vrest - I)/sigma
    f = lambda u: (1/u)*np.exp(-u**2)*(np.exp(2*Yth*u)-np.exp(2*Yr*u))
    quad = scipy.integrate.quad(f, 0, np.inf)[0]
    return 1/(tau*quad)

class ParamsLIF(object):
    def __init__(self, dt = 0.001, tr = 0.003, mu = 1, reset = 0, xsigma = 1, n = 2, tau = 1,\
        c = 0.99, sigma = 20):

        self.dt = dt            #Step size
        self.tr = tr            #Refractory period
        self.mu = mu            #Threshold
        self.reset = reset      #Reset potential
        self.xsigma = xsigma    #Std dev of input x
        self.n = n              #Number of neurons
        self.tau = tau          #Time constant
        self.c = c              #Correlation between noise inputs
        self.sigma = sigma      #Std dev of noise process

class ParamsLIF_Recurrent(object):
    def __init__(self, kernel, dt = 0.001, tr = 0.003, mu = 1, reset = 0, xsigma = 1, n1 = 2, n2 = 10, tau = 1,\
        c = 0.99, sigma = 20):

        self.dt = dt            #Step size
        self.tr = tr            #Refractory period
        self.mu = mu            #Threshold
        self.reset = reset      #Reset potential
        self.xsigma = xsigma    #Std dev of input x
        self.n1 = n1            #Number of neurons in first layer
        self.n2 = n2            #Number of neurons in second layer
        self.n = n1 + n2        #Total number of neurons
        self.tau = tau          #Time constant
        self.c = c              #Correlation between noise inputs
        self.sigma = sigma      #Std dev of noise process
        self.kernel = kernel    #Kernel to apply to spike trains

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
    def __init__(self, params, t = 10, t_total = None):
        self.setup(params, t, t_total)

    def setup(self, params = None, t = None, t_total = None):
        if params is not None:
            self.params = params
        if t is not None:
            self.t = t
        if not hasattr(self, 't_total'):
            self.t_total = None
        if t_total is not None:
            self.t_total = t_total

        #Initialize voltage and spike train variables
        self.T = np.ceil(self.t/self.params.dt).astype(int)
        self.Tr = np.ceil(self.params.tr/self.params.dt).astype(int)
        self.times = np.linspace(0,self.t,self.T)
        
        self.x = 0
        self.W = np.array([5, 5])
        self.V = np.array([8, -4])

        self.keepstate = True
        self.vt = np.zeros(self.params.n)
        self.ut = np.zeros(self.params.n)
        
        if t_total is not None:
            self.T_total = np.ceil(self.t_total/self.params.dt).astype(int)
            #Precompute noise
            self.xi = self.params.sigma*rand.randn(self.params.n+1,self.T_total+self.T)/np.sqrt(self.params.tau)
            self.xi[0,:] = self.xi[0,:]*np.sqrt(self.params.c)
            self.xi[1:,:] = self.xi[1:,:]*np.sqrt(1-self.params.c)
            self.xi_perturb = rand.randn(self.params.n,self.T_total+self.T)/np.sqrt(self.params.tau)
        self.count = 0

    def simulate(self, deltaT = None):

        #if deltaT is provided then in blocks of deltaT we compute the counterfactual trace... the evolution without spiking.
        v = np.zeros((self.params.n,self.T))

        if deltaT is not None:
            u = np.zeros((self.params.n,self.T))
        else:
            u = None

        h = np.zeros((self.params.n,self.T))

        if not self.keepstate:
            self.vt = np.zeros(self.params.n)
            self.ut = np.zeros(self.params.n)

        vt = self.vt
        ut = self.ut

        r = np.zeros(self.params.n)

        #Generate new noise with each sim
        if self.t_total is None:
            xi = self.params.sigma*rand.randn(self.params.n+1,self.T)/np.sqrt(self.params.tau)
            xi[0,:] = xi[0,:]*np.sqrt(self.params.c)
            xi[1:,:] = xi[1:,:]*np.sqrt(1-self.params.c)
        else:
            #Select noise from precomputed noise
            xi = self.xi[:,(self.T*(self.count)):(self.T*(self.count+1))]
        #print xi.shape
        #print self.T

        self.count += 1

        #Simulate t seconds
        for i in range(self.T):

            #ut is not reset by spiking. ut is set to vt at the start of each block of deltaT
            if deltaT is not None:
                if i%deltaT == 0:
                    ut = vt

            #print self.W.shape
            #print xi.shape

            dv = -vt/self.params.tau + np.multiply(self.W,(self.x + xi[0,i] + xi[1:,i]))
            #print vt.shape
            vt = vt + self.params.dt*dv
            ut = ut + self.params.dt*dv
            #Find neurons that spike
            s = vt>self.params.mu
            #print vt.shape
            #Save the voltages and spikes
            h[:,i] = s.astype(int)
            v[:,i] = vt

            if deltaT is not None:
                u[:,i] = ut

            #Make spiking neurons refractory
            r[s] = self.Tr
            #Set the refractory neurons to v_reset
            vt[r>0] = self.params.reset
            vt[vt<self.params.reset] = self.params.reset
            ut[ut<self.params.reset] = self.params.reset
            #Decrement the refractory counters
            r[r>0] -= 1

        #Cost function per time point
        C = (self.V[0]*h[0,:]+self.V[1]*h[1,:]-self.x**2)**2

        #True causal effect for each unit
        beta1 = self.V[0]**2 + 2*self.V[0]*self.V[1]*np.mean(h[1,:])-2*self.V[0]*self.x**2
        beta2 = self.V[1]**2 + 2*self.V[0]*self.V[1]*np.mean(h[0,:])-2*self.V[1]*self.x**2
        betas = [beta1, beta2]

        self.vt = vt

        return (v, h, C, betas, u)

    def simulate_perturbed(self, sigma_perturb):

        v = np.zeros((self.params.n,self.T))
        h = np.zeros((self.params.n,self.T))

        if not self.keepstate:
            self.vt = np.zeros(self.params.n)
            self.ut = np.zeros(self.params.n)

        vt = self.vt

        r = np.zeros(self.params.n)

        if self.T_total is not None:
            #print self.T_total 
            #print self.count 
            if self.count == self.T_total:
                print 'HI'
            self.count = self.count % int(self.T_total/self.T)


        #Generate new noise with each sim
        if self.t_total is None:
            xi = self.params.sigma*rand.randn(self.params.n+1,self.T)/np.sqrt(self.params.tau)
            xi[0,:] = xi[0,:]*np.sqrt(self.params.c)
            xi[1:,:] = xi[1:,:]*np.sqrt(1-self.params.c)
            xi_perturb = sigma_perturb*rand.randn(self.params.n,self.T)/np.sqrt(self.params.tau)
        else:
            #Select noise from precomputed noise
            #The same stim noise for each sim
            xi = self.xi[:,0:self.T]
            #Or update the noise with each sim
            #xi = self.xi[:,(self.T*(self.count)):(self.T*(self.count+1))]
            xi_perturb = sigma_perturb*self.xi_perturb[:,(self.T*(self.count)):(self.T*(self.count+1))]

        #print xi.shape
        #print self.T

        s_input = self.x + np.vstack((xi[0,:],xi[0,:])) + xi[1:,:]

        self.count = self.count + 1
        #print self.count

        #Simulate t seconds
        #print xi.shape
        #print xi_perturb.shape
        for i in range(self.T):
            dv = -vt/self.params.tau + np.multiply(self.W,(self.x + xi[0,i] + xi[1:,i])) + xi_perturb[:,i]
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

        #Keep 
        eligibility = np.sum(np.multiply(s_input, xi_perturb), 1)

        return (v, h, C, betas, eligibility)

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
    def __init__(self, params, t = 1, t_total = 1, nx = 2, no = 1, tau_s = 0.020, onrate = 200, offrate = 100, alpha = 30, spectral_radius = 0.95):
        self.tau_s = tau_s
        self.onrate = onrate
        self.offrate = offrate
        self.alpha = alpha
        self.t_total = t_total
        self.spectral_radius = spectral_radius
        self.setup(params, t, nx, no)

    def setup(self, params = None, t = None, nx = None, no = None, t_total = None):
        #print("I am setting up LIF")
        if params is not None:
            self.params = params
        if t is not None:
            self.t = t
        if nx is not None:
            self.nx = nx
        if no is not None:
            self.no = no
        if t_total is not None:
            self.t_total = t_total

        #Initialize voltage and spike train variables
        self.T = np.ceil(self.t/self.params.dt).astype(int)
        self.Tr = np.ceil(self.params.tr/self.params.dt).astype(int)
        self.times = np.linspace(0,self.t,self.T)
        self.T_total = np.ceil(self.t_total/self.params.dt).astype(int)
        
        self.x = np.ones((self.nx,1))
        self.W = self.alpha*np.random.rand(self.params.n, self.nx)-self.alpha/3.
        self.U = self.alpha*np.random.rand(self.no, self.params.n)-self.alpha/3.

        t_filter = np.linspace(0, 1, 2000)
        exp_filter = np.exp(-t_filter/self.tau_s)
        self.exp_filter = exp_filter/np.sum(exp_filter)
        self.ds = exp_filter[0]

        self.vth = np.zeros(self.params.n)
        self.vto = np.zeros(self.no)

        self.sx = np.zeros((self.nx, self.T_total))
        self.so = np.zeros((self.params.n, self.T_total))

        #Generate noise for the whole sim
        self.xi = self.params.sigma*rand.randn(self.nx+1,self.T_total)/np.sqrt(self.params.tau)
        #Common noise
        self.xi[0,:] = self.xi[0,:]*np.sqrt(self.params.c)
        #Indep noise
        self.xi[1:,:] = self.xi[1:,:]*np.sqrt(1-self.params.c)
        #Output noise
        self.xo = self.params.sigma*rand.randn(self.no,self.T_total)/np.sqrt(self.params.tau)
        #Input noise
        self.xi_i = np.random.rand(self.nx, self.T_total)

        self.Toffset = 0

    def simulate(self, x):

        vh = np.zeros((self.params.n,self.T))
        hh = np.zeros((self.params.n,self.T))
        vo = np.zeros((self.no, self.T))
        ho = np.zeros((self.no, self.T))
        uh = np.zeros((self.params.n,self.T))
        uo = np.zeros((self.no, self.T))

        sx = self.sx
        so = self.so
        Toffset = self.Toffset

        #Take the relevant blocks of precomputed noise
        xi = self.xi[:,Toffset:(Toffset+self.T)]
        xi_i = self.xi_i[:,Toffset:(Toffset+self.T)]
        xo = self.xo[:,Toffset:(Toffset+self.T)]

        #Generate a noisy input process: filtered Poisson spiking
        for i in range(self.nx):
            rate = self.onrate if x[i] == 1 else self.offrate
            #print("x%d = %f"%(i, rate))
            ip = xi_i[i,:] < rate*self.params.dt
            convolve_online(sx[i,:], ip, self.exp_filter, Toffset)

        vt = self.vth
        ut = vt.copy()
        dv = np.zeros(self.params.n)
        r = np.zeros(self.params.n)
        #Simulate t seconds for the hidden layer
        for t in range(self.T):
            dv = -vt/self.params.tau + np.squeeze(np.dot(self.W, sx[:,Toffset+t]+xi[0,t] + xi[1:,t]))
            vt = vt + self.params.dt*dv
            ut = ut + self.params.dt*dv
            #Find neurons that spike
            s = vt>self.params.mu
            #Make spiking neurons refractory
            r[s] = self.Tr
            #Save the voltages and spikes
            hh[:,t] = s.astype(int)
            vh[:,t] = vt
            uh[:,t] = ut
            #Set the refractory neurons to v_reset
            vt[r>0] = self.params.reset
            vt[vt<self.params.reset] = self.params.reset
            ut[ut<self.params.reset] = self.params.reset
            #Decrement the refractory counters
            r[r>0] -= 1
        self.vth = vt

        vt = self.vto
        for i in range(self.params.n):
            convolve_online(so[i,:], hh[i,:], self.exp_filter, Toffset)

        #Simulate t seconds output neuron
        ut = vt.copy()
        dv = np.zeros(self.no)
        r = np.zeros(self.no)
        for t in range(self.T):
            #Determine the output neuron's output
            dv = -vt/self.params.tau + np.squeeze(np.dot(self.U, so[:,Toffset+t] + xo[:,t]))
            vt = vt + self.params.dt*dv
            ut = ut + self.params.dt*dv
            #Find neurons that spike
            s = vt>self.params.mu
            #Make spiking neurons refractory
            r[s] = self.Tr
            ho[:,t] = s.astype(int)
            vo[:,t] = vt
            uo[:,t] = ut
            #Set the refractory neurons to v_reset
            vt[r>0] = self.params.reset
            vt[vt<self.params.reset] = self.params.reset
            ut[ut<self.params.reset] = self.params.reset
            #Decrement the refractory counters
            r[r>0] -= 1
        self.vto = vt

        #Save so and sx for next round
        self.so = so
        self.sx = sx 
        self.Toffset += self.T

        return vh, hh, vo, ho, uh, uo

###################################################################
###################################################################

class LIF_Recurrent(object):

    def __init__(self, params, t = 10, t_total = None):
        self.setup(params, t, t_total)

    def setup(self, params = None, t = None, t_total = None):
        if params is not None:
            self.params = params
        if t is not None:
            self.t = t
        if not hasattr(self, 't_total'):
            self.t_total = None
        if t_total is not None:
            self.t_total = t_total

        #Initialize voltage and spike train variables
        #Simulation time in timestep units
        self.T =  np.ceil(self.t/self.params.dt).astype(int)
        #Refractory period in timestep units
        self.Tr = np.ceil(self.params.tr/self.params.dt).astype(int)
        self.times = np.linspace(0,self.t,self.T)
        
        #Input signal
        self.x = 0
        #Input to each neuron in first layer weights
        self.W1 = 25*np.ones(self.params.n1)
        self.W2 = 2*np.ones(self.params.n2)

        #The feedforward/recurrent connections
        self.U = np.zeros((self.params.n, self.params.n))        
        self.sh = np.zeros((self.params.n, self.T))
        #self.U[0:self.params.n1, self.params.n1:] = 500*np.random.randn(self.params.n1, self.params.n2)
        self.U[self.params.n1:, 0:self.params.n1] = 200*np.random.randn(self.params.n2, self.params.n1)+100

        self.keepstate = True
        self.vt = np.zeros(self.params.n)
        self.ut = np.zeros(self.params.n)
        
        if t_total is not None:
            self.T_total = np.ceil(self.t_total/self.params.dt).astype(int)
            #Precompute noise
            self.xi = self.params.sigma*rand.randn(self.params.n1+1,self.T_total+self.T)/np.sqrt(self.params.tau)
            self.xi_l2 = self.params.sigma*rand.randn(self.params.n2,self.T_total+self.T)/np.sqrt(self.params.tau)
            self.xi[0,:] = self.xi[0,:]*np.sqrt(self.params.c)
            self.xi[1:,:] = self.xi[1:,:]*np.sqrt(1-self.params.c)
            self.xi_perturb = rand.randn(self.params.n,self.T_total+self.T)/np.sqrt(self.params.tau)
        self.count = 0

    def simulate(self, deltaT = None):
        #if deltaT is provided then in blocks of deltaT we compute the counterfactual trace... the evolution without spiking.
        v = np.zeros((self.params.n,self.T))
        if deltaT is not None:
            u = np.zeros((self.params.n,self.T))
        else:
            u = None
        h = np.zeros((self.params.n,self.T))
        if not self.keepstate:
            self.vt = np.zeros(self.params.n)
            self.ut = np.zeros(self.params.n)
        self.sh = np.zeros((self.params.n, self.T))
        vt = self.vt
        ut = self.ut
        sh = self.sh
        r = np.zeros(self.params.n)
        #Generate new noise with each sim
        if self.t_total is None:
            xi = self.params.sigma*rand.randn(self.params.n1+1,self.T)/np.sqrt(self.params.tau)
            xi[0,:] = xi[0,:]*np.sqrt(self.params.c)
            xi[1:,:] = xi[1:,:]*np.sqrt(1-self.params.c)
            xi_l2 = self.params.sigma*rand.randn(self.params.n2,self.T)/np.sqrt(self.params.tau)
        else:
            #Select noise from precomputed noise
            xi = self.xi[:,(self.T*(self.count)):(self.T*(self.count+1))]
            xi_l2 = self.xi_l2[:,(self.T*(self.count)):(self.T*(self.count+1))]
            
        self.count += 1
        #Simulate t seconds
        for i in range(self.T):
            #ut is not reset by spiking. ut is set to vt at the start of each block of deltaT
            if deltaT is not None:
                if i%deltaT == 0:
                    ut = vt
            dv = -vt/self.params.tau + np.dot(self.U,sh[:,i])
            dv[0:self.params.n1] += np.multiply(self.W1,(self.x + xi[0,i] + xi[1:,i]))
            dv[self.params.n1:] += np.multiply(self.W2,(xi_l2[:,i]))
            vt = vt + self.params.dt*dv
            ut = ut + self.params.dt*dv
            #Find neurons that spike
            s = vt>self.params.mu
            #Update sh based on spiking.....
            for s_idx in np.nonzero(s)[0]:
                convolve_online_v2(sh, s_idx, i, self.params.kernel, 0)
            #Save the voltages and spikes
            h[:,i] = s.astype(int)
            v[:,i] = vt
            if deltaT is not None:
                u[:,i] = ut
            #Make spiking neurons refractory
            r[s] = self.Tr
            #Set the refractory neurons to v_reset
            vt[r>0] = self.params.reset
            vt[vt<self.params.reset] = self.params.reset
            ut[ut<self.params.reset] = self.params.reset
            #Decrement the refractory counters
            r[r>0] -= 1

        self.vt = vt
        #self.sh = sh
        return (v, h, u, sh)