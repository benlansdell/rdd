library(RcppCNPy)
library(optrdd)

#Load data from simulations
gauss = npyLoad('./sweeps/Gaussian_1000.npz.npy')
lif = npyLoad('./sweeps/param_w_N_19_i_10_j_10_c_0.010000_deltaT_simulations.npz.npy')

# Simple regression discontinuity with discrete X
threshold = 0
X = gauss
W = as.numeric(X >= threshold)
# using 0.4 for max.second.derivative would have been enough
out.1 = optrdd(X=X, W=W, max.second.derivative = 0.5, estimation.point = threshold)
print(out.1); plot(out.1, xlim = c(-1.5, 1.5))

j = 5
threshold = 1
X = lif[,j]
W = as.numeric(X >= threshold)
# using 0.4 for max.second.derivative would have been enough
out.1 = optrdd(X=X, W=W, max.second.derivative = 3, estimation.point = threshold)
print(out.1); plot(out.1, xlim = c(0, 1.5))

prespike = out.1$gamma.fun.0
postspike = out.1$gamma.fun.1

#Save this back in numpy format...
npySave("optimal_filter_prespike.npy", prespike)
npySave("optimal_filter_postspike.npy", postspike)