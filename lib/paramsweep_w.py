import numpy as np
from lib.lif import LIF, ParamsLIF
from lib.causal import causaleffect

params = ParamsLIF()
lif = LIF(params)

#Simulate for a range of $W$ values.
N = 10
nsims = 500

wvals = np.linspace(1, 10, N)
beta_rd_w = np.zeros((N, N, nsims, params.n))
beta_fd_w = np.zeros((N, N, nsims, params.n))
beta_bp_w = np.zeros((N, N, nsims, params.n))

lif.setup(params)
p = 0.03

for i,w0 in enumerate(wvals):
    for j,w1 in enumerate(wvals):
        print("Running %d simulations with w0=%f, w1=%f"%(nsims, w0, w1))
        lif.W = np.array([w0, w1])
        for k in range(nsims):
            (v, h, Cost, betas) = lif.simulate()
            beta_rd_w[i,j,k,:] = causaleffect(v, Cost, p, params)
            beta_fd_w[i,j,k,:] = causaleffect(v, Cost, 1, params)
            beta_bp_w[i,j,k,:] = betas

#Save output
outfile = './data/output/param_w_N_%d_nsims_%d_default.npz'%(N, nsims)
np.savez(outfile, beta_rd_w=beta_rd_w, beta_fd_w=beta_fd_w, beta_bp_w=beta_bp_w, params = params, wvals = wvals\
	, nsims = nsims, p = p)

#np.loadz(outfile)