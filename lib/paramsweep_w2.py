import numpy as np
from lib.lif import LIF, ParamsLIF
from lib.causal import causaleffect

params = ParamsLIF()
lif = LIF(params)

#Simulate for a range of $W$ values.
N = 20
nsims = 500
wmax = 20
n = params.n 

#Play with different c values
#c = 0.99
c = 0.5

params.c = c
lif.setup(params)

wvals = np.linspace(1, wmax, N)
vs = np.zeros((N, N, nsims, n, lif.T), dtype=np.float16)
hs = np.zeros((N, N, nsims, n, lif.T), dtype=np.bool)

for i,w0 in enumerate(wvals):
    for j,w1 in enumerate(wvals):
        print("Running %d simulations with w0=%f, w1=%f"%(nsims, w0, w1))
        lif.W = np.array([w0, w1])
        for k in range(nsims):
            (v, h, Cost, betas) = lif.simulate()
            vs[i,j,k,:] = v
            hs[i,j,k,:] = h

#Save output
outfile = './sweeps/param_w_N_%d_nsims_%d_c_%f_default_simulations.npz'%(N, nsims, c)
np.savez(outfile, vs = vs, hs = hs, params = params, wvals = wvals\
	, nsims = nsims)

np.savez(outfile, vs = vs, hs = hs, params = params, wvals = wvals\
	, nsims = nsims)
