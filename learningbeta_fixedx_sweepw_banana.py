import numpy as np
from lib.lif import LIF, ParamsLIF, LSM, ParamsLSM, LSM_const

n = 2               # Number of neurons
q = 100             # Number of LSM neurons
x_input = 2         # Constant input
alpha1 = 10         # Cost function params
alpha2 = 30         # Cost function params
tau_s = 0.020       # Time scale for output filter
mu = 1              # Threshold
p = 0.05            # Window size
t = 1               # Time for each epoch
N = 100             # Number of epochs
Wn = 20             # Number of W values we sweep over
wmax = 20           # Max W value of sweep
eta = 1             # Learning rate
#perturb_rate = 0.01 # Proportion of points that are perturbations
#                    # 1% = 10 Hz. Only half of these are spikes, so the injected noise rate is 5Hz

mvals = [0.0025, 0.005, 0.01, 0.015]
M = len(mvals)

# Filename for results
fn_out = './sweeps/learningbeta_fixedx_sweepw_banana_perturbation.npz'

#Cost function params
B1 = 2
B2 = 7
x = .05
y = 0.15
z = -0.2

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
beta_sp = np.zeros((Wn, Wn, n, N, M))

for i, w0 in enumerate(wvals):
    print("W0=%d"%w0)
    for j, w1 in enumerate(wvals):
        #init weights
        lif.W = np.array([w0, w1])
        V = np.ones((n,q))
        U = np.ones((n,q,M))

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
            s_lsm = lsm.simulate(x_input)
            #Simulate LIF
            (v, h, _, _) = lif.simulate()
            s1 = np.convolve(h[0,:], exp_filter)[0:h.shape[1]]
            s2 = np.convolve(h[1,:], exp_filter)[0:h.shape[1]]
        
            abvthr = np.zeros(n)
            blothr = np.zeros(n)

            cost = (B1*s1 - x)**2 + (z + B2*s2 - B2*(2*B1*s1 - y)**2)**2

            ptb = 2*(np.random.rand(*h.shape) < 0.5)-1
            #Create a perturbed set of trains
            for idx2, perturb_rate in enumerate(mvals):
                dU = np.zeros(U.shape[0:2])
                qtb = np.random.rand(*h.shape) < perturb_rate
                h_perturb = h.copy()
                h_perturb[qtb == True] = ptb[qtb == True]
                s1_perturb = np.convolve(h_perturb[0,:], exp_filter)[0:h.shape[1]]
                s2_perturb = np.convolve(h_perturb[1,:], exp_filter)[0:h.shape[1]]
                cost_perturbed = (B1*s1_perturb - x)**2 + (z + B2*s2_perturb - B2*(2*B1*s1_perturb - y)**2)**2
                for t in range(v.shape[1]):
                    for k in range(n):
                        #If this timebin is a perturbation time then update U
                        if qtb[k,t]:
                            dU[k,:] = (np.dot(U[k,:,idx2], s_lsm[:,t])-ptb[k,t]*cost_perturbed[t])*s_lsm[:,t]
                            U[k,:,idx2] = U[k,:,idx2] - eta*dU[k,:]
                s_lsm = lsm.simulate(x_input)
                beta_sp[i,j,:,idx,idx2] = np.mean(np.dot(U[:,:,idx2], s_lsm[:,-100:]),1)

            #cost = (alpha1*s1 + alpha2*s2 - x**2)**2
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
        
            s_lsm = lsm.simulate(x_input)
            beta_rd[i,j,:,idx] = np.mean(np.dot(V, s_lsm[:,-100:]),1)

#Save the results
np.savez(fn_out, wvals = wvals, beta_rd = beta_rd, beta_rd_true = beta_rd_true, beta_fd_true = beta_fd_true,\
 beta_sp = beta_sp)