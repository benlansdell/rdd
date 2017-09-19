import numpy as np
from sklearn import linear_model

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
	lr = linear_model()
	for j in range(v.shape[0]):
		abv = (v[j,:]>mu) & (v[j,:]<(mu+p))
		blo = (v[j,:]<mu) & (v[j,:]>(mu-p))
		C_abv = Cost[abv]
		C_blo = Cost[blo]
		lr.fit(v[j,abv], C_abv) 
		beta_abv = lr.predict(mu)
		lr.fit(v[j,blo], C_blo) 
		beta_blo = lr.predict(mu)
		mce[j] = beta_abv-beta_blo
	return mce