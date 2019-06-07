import numpy as np
from sklearn import linear_model
import copy
import numpy.random as random

#	#Load filter
#	a = np.load(fn_filter)
#	filter = a['filter']

#def causaleffect_optimalized(v, Cost, p, params, filter):
#	mu = params.mu;
#	mce = np.zeros(v.shape[0])
#	for j in range(v.shape[0]):
#		filtered = ?
#		mce[j] = np.mean(np.multiply(Cost, filtered))
#	return mce

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
	lr = linear_model.LinearRegression()
	for j in range(v.shape[0]):
		abv = (v[j,:]>mu) & (v[j,:]<(mu+p))
		blo = (v[j,:]<mu) & (v[j,:]>(mu-p))
		C_abv = Cost[abv]
		C_blo = Cost[blo]
		lr.fit(v[j,abv].reshape((-1,1)), C_abv) 
		beta_abv = lr.predict(mu)
		lr.fit(v[j,blo].reshape((-1,1)), C_blo) 
		beta_blo = lr.predict(mu)
		mce[j] = beta_abv-beta_blo
	return mce

def causaleffect_maxv_linear(v, Cost, deltaT, p, params):
	#Split into blocks of DeltaT (provided in units of seconds)
	cost_r = Cost.reshape((-1, deltaT))
	Cost = np.squeeze(cost_r[:,-1])

	mu = params.mu;
	mce = np.zeros(v.shape[0])

	lr = linear_model.LinearRegression()

	for j in range(v.shape[0]):
		#Take max voltage in each block
		v_r = v[j,:].reshape((-1, deltaT))
		vb = np.max(v_r, 1)

		abv = (vb>mu) & (vb<(mu+p))
		blo = (vb<mu) & (vb>(mu-p))
		C_abv = Cost[abv]
		C_blo = Cost[blo]

		lr.fit(vb[abv].reshape((-1,1)), C_abv) 
		beta_abv = lr.predict(mu)
		lr.fit(vb[blo].reshape((-1,1)), C_blo) 
		beta_blo = lr.predict(mu)

		mce[j] = beta_abv-beta_blo

	return mce

def causaleffect_maxv(v, Cost, deltaT, p, params):
	#Split into blocks of DeltaT (provided in units of seconds)
	cost_r = Cost.reshape((-1, deltaT))
	Cost = np.squeeze(cost_r[:,-1])

	mu = params.mu;
	mce = np.zeros(v.shape[0])

	for j in range(v.shape[0]):
		#Take max voltage in each block
		v_r = v[j,:].reshape((-1, deltaT))
		vb = np.max(v_r, 1)

		abv = (vb>mu) & (vb<(mu+p))
		blo = (vb<mu) & (vb>(mu-p))
		C_abv = Cost[abv]
		C_blo = Cost[blo]
		print('Above', np.sum(abv))
		print('Below', np.sum(blo))

		mce[j] = np.mean(C_abv)-np.mean(C_blo)

	return mce

def causaleffect_maxv_sp(v, h, cost, deltaT, params, exp_filter):
		ace_sp = np.zeros(params.n)
		p = copy.copy(params)
		p.n = 1

		s1 = np.convolve(h[0,:], exp_filter)[0:h.shape[1]]
		s2 = np.convolve(h[1,:], exp_filter)[0:h.shape[1]]
		s = np.vstack((s1, s2))

		for i in range(params.n):
			si = s.copy()
			hp = h.copy()
			vp = v.copy()

			#Perturb vi and hi
			nB = h.shape[1]/deltaT
			sp = random.rand(nB) < 0.5
			vi = 1.1*(sp == True)[None, :]
			hi = np.zeros((nB, deltaT))

			#Choose blocks to randomly spike in
			for b in range(nB):
				if sp[b]:
					idx = np.random.choice(deltaT)
					hi[b, idx] = 1

			hi = np.squeeze(hi.reshape((1,-1)))
			s[i] = np.convolve(hi, exp_filter)[0:h.shape[1]]

			#Compute perturbed cost function 
			Cost = cost(s[0], s[1])
			cost_r = Cost.reshape((-1, deltaT))
			Cost = np.squeeze(cost_r[:,-1])

			#print vi.shape
			#print Cost.shape

			#Then run causaleffect_linear with large window size
			ace_sp[i] = causaleffect(vi, Cost, 1.1, p)

		return ace_sp