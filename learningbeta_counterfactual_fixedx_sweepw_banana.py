import numpy as np
from lib.lif import LIF, ParamsLIF

cvals = [0.01, 0.25, 0.5, 0.75, 0.99]

for c in cvals:

    fn_out = './sweeps/learningbeta_counterfactual_fixedx_sweepw_banana_perturbation_c_%f.npz'%c

    q = 3                   #Dimension of learnt vector
    dt = 0.001              #Simulation timestep
    DeltaT = 20             #Number of timebins over which learning rule is applied
    tsim = 60               #Total simulation time
    T = (tsim/dt)/DeltaT    #Number of learning blocks
    N = 50                  #Number of repeated simulations
    x_input = 0                   #Input
    n = 2                   #Number of neurons
    sigma = 10              #Their noise level
    mu = 1                  #Threshold
    tau = 1                 #Neuron timescale
    eta = 1                #Learning rate
    p = 0.1                 #Learning window
    tau_s = 0.20            #Output filter timescale
    wmin = 2                #Minimum unit weight
    wmax = 20               #Maximum unit weight
    Wn = 19                 #Number of weights to range over
    
    #Cost function
    B1 = 1
    B2 = 2
    x = .01
    y = 0.1
    z = 0
    
    nsims = 50
    
    cost_fun = lambda s1, s2: (B1*s1-x)**2 + (z+B2*s2 - B2*(B1*s1-y)**2)**2
    
    #perturb_rate = 0.01 # Proportion of points that are perturbations
    #                    # 1% = 10 Hz. Only half of these are spikes, so the injected noise rate is 5Hz
    
    mvals = [0.0025, 0.005, 0.01, 0.015]
    M = len(mvals)
        
    params_lif = ParamsLIF(sigma = sigma, tau = tau, mu = mu, c = c)
    lif = LIF(params_lif, t = tsim)
    lif.x = x_input
    
    t_filter = np.linspace(0, 1, 2000)
    exp_filter = np.exp(-t_filter/tau_s)
    exp_filter = exp_filter/np.sum(exp_filter)
    ds = exp_filter[0]
    
    wvals = np.linspace(wmin, wmax, Wn)
    
    beta_rd = np.zeros((Wn, Wn, n, T, nsims))
    beta_rd_mean = np.zeros((Wn, Wn, n, T, nsims))
    beta_rd_true = np.zeros((Wn, Wn, n, T, nsims))
    beta_fd_true = np.zeros((Wn, Wn, n, T, nsims))
    beta_sp = np.zeros((Wn, Wn, n, T, M, nsims))
    
    for i, w0 in enumerate(wvals):
        for j, w1 in enumerate(wvals):
            print("W0=%d, W1=%d"%(w0, w1))
            for ksim in range(nsims):
    
                c1_abv_p = np.zeros(0)
                c1_abv_1 = np.zeros(0)
                c1_blo_p = np.zeros(0)
                c1_blo_1 = np.zeros(0)
                
                c2_abv_p = np.zeros(0)
                c2_abv_1 = np.zeros(0)
                c2_blo_p = np.zeros(0)
                c2_blo_1 = np.zeros(0)
        
                #init weights
                lif.W = np.array([w0, w1])
        
                count = np.zeros(n)
                #Simulate LIF
                (v_raw, h_raw, _, _, u_raw) = lif.simulate(DeltaT)
                s1 = np.convolve(h_raw[0,:], exp_filter)[0:h_raw.shape[1]]
                s2 = np.convolve(h_raw[1,:], exp_filter)[0:h_raw.shape[1]]
                
                V = np.zeros((n, q))
                V_mean = np.zeros((n, q))
                nB = h_raw.shape[1]/DeltaT
                
                abvthr = np.zeros(n)
                blothr = np.zeros(n)
        
                #Create a perturbed set of trains
                for idx2, perturb_rate in enumerate(mvals):
                    U = np.zeros((n, q))
                    dU = np.zeros(U.shape)
        
                    #Perturb spike trains
                    ptb = 2*(np.random.rand(*h_raw.shape) < 0.5)-1
                    qtb = np.random.rand(*h_raw.shape) < perturb_rate
                    h_perturb = h_raw.copy()
                    h_perturb[qtb == True] = ptb[qtb == True]
                    s1_perturb = np.convolve(h_perturb[0,:], exp_filter)[0:h_perturb.shape[1]]
                    s2_perturb = np.convolve(h_perturb[1,:], exp_filter)[0:h_perturb.shape[1]]
                    cost_perturbed = cost_fun(s1_perturb, s2_perturb)
        
                    #Reblock them and run learning rule
                    hp = h_perturb.reshape((n, T, DeltaT))
                    h_perturb = np.max(hp, 2)
                    costp = cost_perturbed.reshape((nB, DeltaT))
                    cost_perturbed = np.squeeze(costp[:,-1])
                    qp = qtb.reshape((n, nB, DeltaT))
                    qtb = np.max(qp, 2)
                    pp = ptb.reshape((n, nB, DeltaT))
                    ptb = np.max(pp, 2)
        
                    for t in range(nB):
                        for k in range(n):
                            #If this timebin is a perturbation time then update U
                            if qtb[k,t]:
                                ahat = np.array([1, 0, 0])
                                #ahat = np.array([1, (v[k,t]-mu), 0])                        
                                dU[k,:] = (np.dot(U[k,:], ahat)-ptb[k,t]*cost_perturbed[t])*ahat
                                U[k,:] = U[k,:] - eta*dU[k,:]
                        beta_sp[i,j,:,t,idx2,ksim] = U[:,0]
                
                cost_raw = cost_fun(s1, s2)
                #Break the simulation and voltage into blocks
                hm = h_raw.reshape((n, nB, DeltaT))
                vm = u_raw.reshape((n, nB, DeltaT))
                
                v = np.max(vm, 2)
                h = np.max(hm, 2)
                cost_r = cost_raw.reshape((nB, DeltaT))
                cost = np.squeeze(cost_r[:,-1])
        
                #Then just repeat the learning rule as before
                dV = np.zeros(V.shape)
                bt = [False, False]
                for t in range(nB):
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
                                #ahat = np.array([1, 0, -(v[k,t]-mu)])
                                ahat = np.array([1, 0, 0])
                                dV[k,:] += (np.dot(V[k,:], ahat)+cost[t])*ahat                    
                                bt[k] = True
                        elif (v[k,t] < mu + p) & (v[k,t] >= mu):
                            if k == 0:
                                c1_abv_p = np.hstack((c1_abv_p, cost[t]))
                            else:
                                c2_abv_p = np.hstack((c2_abv_p, cost[t]))
                            abvthr[k] += 1
                            #Only do the update when firing...
                            if bt[k] == True:
                                #ahat = np.array([1, (v[k,t]-mu), 0])
                                ahat = np.array([1, 0, 0])
                                dV[k,:] += (np.dot(V[k,:], ahat)-cost[t])*ahat                                        
                                count[k] += 1
                                V_mean[k,:] = (V_mean[k,:]*count[k] - dV[k,:])/(count[k]+1)
                                V[k,:] = V[k,:] - eta*dV[k,:]/(count[k]+1)
                                dV[k,:] = np.zeros((1,q))
                                bt[k] = False
                            
                        beta_rd[i,j,k,t,ksim] = V[k,0]
                        beta_rd_mean[i,j,k,t,ksim] = V_mean[k,0]
        
                    beta_rd_true[i,j,0,t,ksim] = np.mean(c1_abv_p)-np.mean(c1_blo_p)
                    beta_rd_true[i,j,1,t,ksim] = np.mean(c2_abv_p)-np.mean(c2_blo_p)
                    beta_fd_true[i,j,0,t,ksim] = np.mean(c1_abv_1)-np.mean(c1_blo_1)
                    beta_fd_true[i,j,1,t,ksim] = np.mean(c2_abv_1)-np.mean(c2_blo_1) 
    
    #Save the results
    np.savez(fn_out, wvals = wvals, beta_rd = beta_rd, beta_rd_true = beta_rd_true, beta_fd_true = beta_fd_true,\
     beta_sp = beta_sp, beta_rd_mean = beta_rd_mean)