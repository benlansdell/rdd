import numpy as np
from lib.lif import LIF, ParamsLIF, LSM, ParamsLSM, LSM_const

n = 2               # Number of neurons
q = 100             # Number of LSM neurons
x = 2               # Constant input
alpha1 = 10         # Cost function params
alpha2 = 30         # Cost function params
tau_s = 0.020       # Time scale for output filter
mu = 1              # Threshold
p = 0.05            # Window size
t = 1               # Time for each epoch
N = 20              # Number of epochs
Wn = 20             # Number of W values we sweep over
wmax = 20           # Max W value of sweep
eta = .5            # Learning rate

# Filename for results
fn_out = './sweeps/learningbeta_fixedx_sweepw.npz'

params = ParamsLSM(q = q, p = 1, t = t)
lsm = LSM(params)
params_lif = ParamsLIF()
lif = LIF(params_lif, t = t)

t_filter = np.linspace(0, 0.15, 150)
exp_filter = np.exp(-t_filter/tau_s)
exp_filter = exp_filter/np.sum(exp_filter)
ds = exp_filter[0]

wvals = np.linspace(1, wmax, Wn)
beta_rd = np.zeros((Wn, Wn, n, N))
beta_rd_true = np.zeros((Wn, Wn, n, N))
beta_fd_true = np.zeros((Wn, Wn, n, N))

for i, w0 in enumerate(wvals):
    print("W0=%d"%w0)
    for j, w1 in enumerate(wvals):
        #init weights
        lif.W = np.array([w0, w1])
        V = np.ones((n,q))

        #Also collect the c_abv, c_below for p = 0.03, p = 1, accumulated over each epoch, and estimate 
        #the 'true' beta as we go
        c1_abv_p = np.zeros(0)
        c1_abv_1 = np.zeros(0)
        c1_blo_p = np.zeros(0)
        c1_blo_1 = np.zeros(0)
        
        c2_abv_p = np.zeros(0)
        c2_abv_1 = np.zeros(0)
        c2_blo_p = np.zeros(0)
        c2_blo_1 = np.zeros(0)
        
        count = 0

        for idx in range(N):
            #Simulate LSM
            s_lsm = lsm.simulate(x)
            #Simulate LIF
            (v, h, _, _) = lif.simulate()
            s1 = np.convolve(h[0,:], exp_filter)[0:h.shape[1]]
            s2 = np.convolve(h[1,:], exp_filter)[0:h.shape[1]]
        
            dVabv = np.zeros(V.shape)
            dVblo = np.zeros(V.shape)
            
            abvthr = np.zeros(n)
            blothr = np.zeros(n)
            
            cost = (alpha1*s1 + alpha2*s2 - x**2)**2
            dV = np.zeros(V.shape)
            bt = [False, False]
            for t in range(v.shape[1]):
                for k in range(n):
                    if (v[k,t] < mu):
                        if k == 0:
                            c1_blo_1 = np.hstack((c1_blo_1, cost[t]))
                        else:
                            c2_blo_1 = np.hstack((c2_blo_1, cost[t]))
                    if (v[k,t] >= mu):
                        if k == 0:
                            c1_abv_1 = np.hstack((c1_abv_1, cost[t]))
                        else:
                            c2_abv_1 = np.hstack((c2_abv_1, cost[t]))
                            
                    if (v[k,t] > mu - p) & (v[k,t] < mu):
                        if k == 0:
                            c1_blo_p = np.hstack((c1_blo_p, cost[t]))
                        else:
                            c2_blo_p = np.hstack((c2_blo_p, cost[t]))
                        blothr[k] += 1
                        if bt[k] == False:
                            dV[k,:] += (np.dot(V[k,:], s_lsm[:,t])+cost[t])*s_lsm[:,t]
                            bt[k] = True
                    elif (v[k,t] < mu + p) & (v[k,t] >= mu):
                        if k == 0:
                            c1_abv_p = np.hstack((c1_abv_p, cost[t]))
                        else:
                            c2_abv_p = np.hstack((c2_abv_p, cost[t]))
                        abvthr[k] += 1
                        #Only do the update when firing...
                        if bt[k] == True:
                            dV[k,:] += (np.dot(V[k,:], s_lsm[:,t])-cost[t])*s_lsm[:,t]
                            count += 1
                            V[k,:] = V[k,:] - eta*dV[k,:]#*N/(N+1)
                            dV[k,:] = np.zeros((1,q))
                            bt[k] = False
            
            beta_rd_true[i,j,0,idx] = np.mean(c1_abv_p)-np.mean(c1_blo_p)
            beta_rd_true[i,j,1,idx] = np.mean(c2_abv_p)-np.mean(c2_blo_p)
            beta_fd_true[i,j,0,idx] = np.mean(c1_abv_1)-np.mean(c1_blo_1)
            beta_fd_true[i,j,1,idx] = np.mean(c2_abv_1)-np.mean(c2_blo_1)
        
            s_lsm = lsm.simulate(x)
            beta_rd[i,j,:,idx] = np.mean(np.dot(V, s_lsm[:,-100:]),1)

#Save the results
np.savez(fn_out, wvals = wvals, beta_rd = beta_rd, beta_rd_true = beta_rd_true, beta_fd_true = beta_fd_true)