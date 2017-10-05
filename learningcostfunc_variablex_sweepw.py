import numpy as np
import numpy.random as rand
from lib.lif import LIF, ParamsLIF, LSM, ParamsLSM, LSM_const

n = 2               # Number of neurons
tau_s = 0.060       # Time scale for output filter
mu = 1              # Threshold
t = 1               # Time for each epoch
nsims = 1000         # Number of epochs
N = 30              # Number of W values we sweep over
wmax = 8           # Max W value of sweep
wmin = -8          # Min w value of sweep
eta = .5            # Learning rate
sigma = 3           # Spread of input x distribution
c = 0.75            # Noise correlation coefficient

# Filename for results
fn_out = './sweeps/learningcostfunc_variablex_sweepw.npz'

params_lif = ParamsLIF()
params_lif.c = c
lif = LIF(params_lif, t = t)

t_filter = np.linspace(0, 0.45, 450)
exp_filter = np.exp(-t_filter/tau_s)
exp_filter = exp_filter/np.sum(exp_filter)
ds = exp_filter[0]

wvals = np.linspace(wmin, wmax, N)
vs = np.zeros((N, N, nsims, n, lif.T), dtype=np.float16)
hs = np.zeros((N, N, nsims, n, lif.T), dtype=np.bool)
ss = np.zeros((N, N, nsims, n, lif.T), dtype=np.float32)
xs = np.zeros((N, N, nsims))

for i, w0 in enumerate(wvals):
    print("W0=%d"%w0)
    for j, w1 in enumerate(wvals):
        #init weights
        lif.W = np.array([w0, w1])
        for k in range(nsims):
            x = sigma*rand.randn()
            lif.x = x
            (v, h, _, _) = lif.simulate()
            s1 = np.convolve(h[0,:], exp_filter)[0:h.shape[1]]
            s2 = np.convolve(h[1,:], exp_filter)[0:h.shape[1]]
 
            xs[i,j,k] = x
            vs[i,j,k,:] = v
            hs[i,j,k,:] = h
            ss[i,j,k,0,:] = s1
            ss[i,j,k,1,:] = s2
               
#Save the results
np.savez(fn_out, ss = ss, vs = vs, hs = hs, xs = xs, params = params_lif, wvals = wvals, nsims = nsims)