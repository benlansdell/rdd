import numpy as np
from lib.lif import LIF, ParamsLIF
from lib.causal import causaleffect

#Set x = 0, sigma = 10
#wvals = 2..20
sigma = 10
mu = 1
tau = 1
t = 500
params = ParamsLIF(sigma = sigma, mu = mu, tau = tau)
lif = LIF(params, t = t)
lif.x = 0

#Simulate for a range of $W$ values.
N = 19
nsims = 1
wmax = 20
n = params.n
deltaT = 50

#Play with different c values
cvals = [0.01, 0.25, 0.5, 0.75, 0.99]

for c in cvals:
	print("Running simulations for c = %f"%c)
	outfile = './data/output/param_w_N_%d_nsims_%d_c_%f_deltaT_counterfactual_simulations.npz'%(N, nsims, c)
	params.c = c
	lif.setup(params)
	lif.x = 0
	wvals = np.linspace(2, wmax, N)
	vs = np.zeros((N, N, nsims, n, lif.T), dtype=np.float16)
	hs = np.zeros((N, N, nsims, n, lif.T), dtype=np.bool)
	us = np.zeros((N, N, nsims, n, lif.T), dtype=np.float16)
	for i,w0 in enumerate(wvals):
	    for j,w1 in enumerate(wvals):
	        print("Running %d simulations with w0=%f, w1=%f"%(nsims, w0, w1))
	        lif.W = np.array([w0, w1])
	        for k in range(nsims):
	            (v, h, Cost, betas, u) = lif.simulate(deltaT)
	            vs[i,j,k,:] = v
	            hs[i,j,k,:] = h
	            us[i,j,k,:] = u
	#Save output
	np.savez(outfile, vs = vs, hs = hs, params = params, wvals = wvals\
		, nsims = nsims, us = us)