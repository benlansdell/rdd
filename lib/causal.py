import numpy as np

def causaleffect(v, Cost, p, params):
	mu = params.mu;
	mce = np.zeros(v.shape[0])
	for j in range(v.shape[0]):
		abv = (v[j,:]>mu) & (v[j,:]<(mu+p))
		blo = (v[j,:]<mu) & (v[j,:]>(mu-p))
		C_abv = Cost[abv]
		C_blo = Cost[blo]
		mce[j] = np.mean(C_abv)-np.mean(C_blo)
	return mce

def causaleffect_linear(v, Cost, p, params):
	mu = params.mu;
	mce = np.zeros(v.shape[0])
	for j in range(v.shape[0]):
		abv = (v[j,:]>mu) & (v[j,:]<(mu+p))
		blo = (v[j,:]<mu) & (v[j,:]>(mu-p))
		C_abv = Cost[abv]
		C_blo = Cost[blo]
		mce[j] = np.mean(C_abv)-np.mean(C_blo)
	return mce