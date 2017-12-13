import sys
import numpy as np
import numpy.random as rand
from lib.lif import LIF, ParamsLIF
from lib.mu_w_2_20 import mu_mean

#Things we vary per run
etas = [1, 1e-1, 1e-2]     #Cost gradient learning rate (RDD)
#eta = etas[0]

#Keep constant
decaystepsize = False
c = 0.01

#Change these
linearcorrection = False
p = .1

for eta in etas:

	flags = 'eta_%f_decaystep_%d_lincorr_%d_c_%f_p_%f'%(eta, decaystepsize,linearcorrection,c, p)
	
	#Starting weights
	wmax = 20
	wmin = 2
	N = 10
	wvals = np.linspace(wmin, wmax, N)
	
	#Things we don't vary per run
	q = 3                      #Dimension of learnt vector
	dt = 0.001                 #Simulation timestep
	DeltaT = 50                #Number of timebins over which learning rule is applied
	tsim = 500                 #Total simulation time
	T = int((tsim/dt)/DeltaT)  #Number of learning blocks
	Nsims = 10                 #Number of repeated simulations
	#c = 0.01                   #Correlation coefficient
	x_input = 0                #Input
	n = 2                      #Number of neurons
	sigma = 10                 #Their noise level
	mu = 1                     #Threshold
	tau = 1                    #Neuron timescale
	epsilon = 2e5              #Weight learning rate (RDD)
	#p = 0.5                    #Learning window
	tau_s = 0.20               #Output filter timescale
	Nmu = 19				   #Number of components of the mu activity function
	
	#Cost function parameters
	B1 = 1
	B2 = 2
	x = .01
	y = 0.1
	z = 0
	cost_fun = lambda s1, s2: (B1*s1-x)**2 + (z+B2*s2 - B2*(B1*s1-y)**2)**2
	
	def convolve_online(s, h, kernel, t_offset):
		#print np.nonzero(h)[0]
		for idx in np.nonzero(h)[0]:
			st = t_offset + idx
			en = min(s.shape[0], st + kernel.shape[0])
			ln = en-st
			#print st,en,ln,idx,t_offset
			s[st:en] += kernel[0:ln]
		
	t_filter = np.linspace(0, 1, 2000)
	exp_filter = np.exp(-t_filter/tau_s)
	exp_filter = exp_filter/np.sum(exp_filter)
	ds = exp_filter[0]
	
	fn_out = './sweeps/learningw_counterfactual_flags_%s.npz'%flags
	
	W_rdd_trace = np.zeros((N, N, n, T, Nsims))
	V_trace = np.zeros((N, N, n, T, Nsims))
	
	for idx1, w1 in enumerate(wvals):
		for idx2, w2 in enumerate(wvals):
	
			W_rdd = np.array([w1,w2], 'float64')
			
			sys.stdout.write("W1 = %d, W2 = %d"%(w1, w2))
			sys.stdout.flush()
	
			#Setup neurons
			params_lif = ParamsLIF(sigma = sigma, tau = tau, mu = mu, c = c)
			lif_rdd = LIF(params_lif, t = DeltaT*dt, t_total = tsim)
			lif_rdd.W = W_rdd.copy()
			lif_rdd.x = x_input		
			
			for idx in range(Nsims):
				#Create the filtered output vectors now, fill them as we go
				s1 = np.zeros(int(T/dt))
				s2 = np.zeros(int(T/dt))
				
				lif_rdd = LIF(params_lif, t = DeltaT*dt, t_total = tsim)
				lif_rdd.W = W_rdd.copy()
				lif_rdd.x = x_input
			
				count = np.zeros(n)
				sys.stdout.write('.')
				sys.stdout.flush()
			
				lif_rdd.count = 0
				bt = [False, False]
			
				V = np.zeros((n, q))
				#Then just repeat the learning rule as before
				dV = np.zeros(V.shape)
			
				for j in range(T):
					#Simulate LIF for RDD
					#print("t = %d"%j)
					(v_raw, h_raw, _, _, u_raw) = lif_rdd.simulate(DeltaT)
			
					t_offset = j*DeltaT
					convolve_online(s1, h_raw[0,:], exp_filter, t_offset)
					convolve_online(s2, h_raw[1,:], exp_filter, t_offset)
					cost = cost_fun(s1[t_offset+DeltaT], s2[t_offset+DeltaT])
					nB = h_raw.shape[1]/DeltaT 
					um = u_raw.reshape((n, nB, DeltaT))
					u = np.max(um, 2)
					
					for k in range(n):
						if (u[k,0] > mu - p) & (u[k,0] < mu):
							if bt[k] == False:
								if linearcorrection == True:
									ahat = np.array([1, 0, -(u[k,0]-mu)])
								else:
									ahat = np.array([1, 0, 0])
								dV[k,:] += (np.dot(V[k,:], ahat)+cost)*ahat                    
								bt[k] = True
						elif (u[k,0] < mu + p) & (u[k,0] >= mu):
							#Only do the update when firing...
							if bt[k] == True:
								if linearcorrection == True:
									ahat = np.array([1, (u[k,0]-mu), 0])
								else:
									ahat = np.array([1, 0, 0])
								dV[k,:] += (np.dot(V[k,:], ahat)-cost)*ahat                                        
								count[k] += 1
								if decaystepsize == True:
									V[k,:] = V[k,:] - eta*dV[k,:]/(count[k]+1)
								else:
									V[k,:] = V[k,:] - eta*dV[k,:]
								dV[k,:] = np.zeros((1,q))
								bt[k] = False
							
					#At end of episode, update weights according to V for RDD
					lif_rdd.W -= epsilon*np.multiply(V[:,0], mu_mean(lif_rdd.W, wmin, wmax, Nmu))#/(j+100)
					lif_rdd.W = np.maximum(np.minimum(lif_rdd.W, wmax), wmin)
					W_rdd_trace[idx1, idx2, :,j,idx] = lif_rdd.W
					V_trace[idx1, idx2, :,j,idx] = V[:,0]
	
			sys.stdout.write('\n')
			sys.stdout.flush()
	
	#Save the learning traces
	np.savez(fn_out, W_rdd_trace = W_rdd_trace, V_trace = V_trace)