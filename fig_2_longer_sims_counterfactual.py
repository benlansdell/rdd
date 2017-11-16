import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand
import pandas as pd
import seaborn as sns

from lib.lif import LIF, ParamsLIF
from lib.causal import causaleffect_maxv, causaleffect_maxv_linear, causaleffect_maxv_sp

fn_out = './sweeps/fig2_longer_sims_counterfactual.npz'

nsims = 1000
cvals = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
pvals = np.linspace(0.02, 1, 10)

tau_s = 0.020
dt = 0.001
t = 50

alpha1 = -30
alpha2 = 20
x = 2

DeltaT = 50

params = ParamsLIF()
params_orig = ParamsLIF()
lif = LIF(params, t = t)

t_filter = np.linspace(0, 0.150, 150)
exp_filter = np.exp(-t_filter/tau_s)
exp_filter = exp_filter/np.sum(exp_filter)
ds = exp_filter[0]

#c (correlation between noise inputs)
beta_rd_c = np.zeros((len(cvals), nsims, len(pvals), params.n))
beta_fd_c = np.zeros((len(cvals), nsims, params.n))
beta_bp_c = np.zeros((len(cvals), nsims, params.n))

beta_rd_c_linear = np.zeros((len(cvals), nsims, len(pvals), params.n))
beta_fd_c_linear = np.zeros((len(cvals), nsims, params.n))

m_beta_bp_c = np.zeros(params.n)
m_beta_rd_c = np.zeros((len(cvals), params.n))
se_beta_fd_c = np.zeros((len(cvals), params.n))
m_beta_rd_c = np.zeros((len(cvals), params.n))
se_beta_fd_c = np.zeros((len(cvals), params.n))

m_beta_rd_c_linear = np.zeros((len(cvals), params.n))
se_beta_fd_c_linear = np.zeros((len(cvals), params.n))
m_beta_rd_c_linear = np.zeros((len(cvals), params.n))
se_beta_fd_c_linear = np.zeros((len(cvals), params.n))

beta_sp_c = np.zeros((len(cvals), params.n))
cost = lambda s1, s2: (alpha1*s1 + alpha2*s2 - x**2)**2

for i,c in enumerate(cvals):
    print("Running %d simulations with c=%s"%(nsims, c))
    params.c = c
    lif.setup(params)
    for j in range(nsims):
        (v, h, _, _, u) = lif.simulate(DeltaT)
        s1 = np.convolve(h[0,:], exp_filter)[0:h.shape[1]]
        s2 = np.convolve(h[1,:], exp_filter)[0:h.shape[1]]
        cost_s = cost(s1, s2)
        for k,p in enumerate(pvals):
            #print("p = %f"%p)
            beta_rd_c[i,j,k,:] = causaleffect_maxv(u, cost_s, DeltaT, p, params)
            beta_rd_c_linear[i,j,k,:] = causaleffect_maxv_linear(u, cost_s, DeltaT, p, params)
        beta_fd_c[i,j,:] = causaleffect_maxv(u, cost_s, DeltaT, 1, params)
        beta_fd_c_linear[i,j,:] = causaleffect_maxv_linear(u, cost_s, DeltaT, 1, params)
        #Compute the SP cost
        beta_sp_c[i,:] = causaleffect_maxv_sp(u, h, cost, DeltaT, params, exp_filter)        

m_beta_rd_c = np.mean(beta_rd_c, 1)
se_beta_rd_c = np.std(beta_rd_c, 1)
m_beta_fd_c = np.mean(beta_fd_c, 1)
se_beta_fd_c = np.std(beta_fd_c, 1)

m_beta_rd_c_linear = np.mean(beta_rd_c_linear, 1)
se_beta_rd_c_linear = np.std(beta_rd_c_linear, 1)
m_beta_fd_c_linear = np.mean(beta_fd_c_linear, 1)
se_beta_fd_c_linear = np.std(beta_fd_c_linear, 1)

#Save the results
np.savez(fn_out, wvals = cvals, beta_rd_c = beta_rd_c, beta_fd_c = beta_fd_c, beta_rd_c_linear = beta_rd_c_linear, beta_fd_c_linear = beta_fd_c_linear, beta_sp_c = beta_sp_c)

u = 0
fig,ax = plt.subplots(1,1,figsize=(4,4))
for i in range(len(cvals)):
    sns.tsplot(data = beta_rd_c[i,:,:,u], ax = ax, ci='sd', time=pvals, color='C%d'%i)
    #sns.tsplot(data = beta_rd_c[i,:,:,u], ax = ax, time=pvals, color='C%d'%i)
    ax.plot(pvals, m_beta_rd_c[i,-1,u]*ones(pvals.shape), '-.', color='C%d'%i)
ax.set_xlabel('window size $p$');
ax.set_ylabel('average causal effect');
ax.set_title('Constant RD estimator');
ax.plot(pvals, beta_sp_c[0,u]*ones(pvals.shape), color=(0,0,0));
ax.set_ylim([1.8, 6])
ax.set_xlim([0, 1])
ax.set_yticks([2, 3, 4, 5, 6])
sns.despine(trim=True)
ax.legend(["c = %.2f"%i for i in cvals]);
plt.savefig('./fig_2a.pdf')

fig,ax = plt.subplots(1,1,figsize=(4,4))
for i in range(len(cvals)):
    sns.tsplot(data=beta_rd_c_linear[i,:,:,u], ax = ax, ci='sd', time=pvals, color='C%d'%i)
    #sns.tsplot(data=beta_rd_c_linear[i,:,:,u], ax = ax, time=pvals, color='C%d'%i)
    ax.plot(pvals, m_beta_rd_c_linear[i,-1,u]*ones(pvals.shape), '-.', color='C%d'%i)
ax.set_xlabel('window size $p$');
ax.set_ylabel('average causal effect');
ax.set_title('Linear RD estimator');
ax.plot(pvals, beta_sp_c[0,u]*ones(pvals.shape), color=(0,0,0));
ax.set_ylim([1.8, 6])
ax.set_xlim([0, 1])
ax.set_yticks([2, 3, 4, 5, 6])
sns.despine(trim=True)
plt.savefig('./fig_2a_linear.pdf')