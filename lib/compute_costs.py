from lib.lif import ParamsLIF

params = ParamsLIF()

#Load simulations and compute different average cost functions
N = 20
nsims = 500
c = 0.99

fn_in = './data/output/param_w_N_%d_nsims_%d_c_%f_default_simulations.npz'%(N, nsims, c)
sims = np.load(outfile, vs = vs, hs = hs, params = params, wvals = wvals\
	, nsims = nsims)

#Compute cost
v1 = 4
v2 = 5
x = 20

tau_s = 0.03
dt = 0.001

costs2 = np.zeros((N,N))
costs3 = np.zeros((N,N))

costs2_filtered = np.zeros((N,N))
costs3_filtered = np.zeros((N,N))

for i in range(N):
	for j in range(N):
		h1 = hs[i,j,:,0,:]
		h2 = hs[i,j,:,1,:]
		c2 = (v1*1000*h1 + v2*1000*h2 - x**2)**2
		c3 = (h1 - 0.1*x**2)**2 + (h2-.02*(h1-70)**2-60)**2

		#c2 = (v1*1000*h1 + v2*1000*h2 - x**2)**2
		#c3 = (h1*1000 - 0.1*x**2)**2 + (h2*1000-.02*(h1*1000-70)**2-60)**2

		costs2[i,j] = np.mean(c2)
		costs3[i,j] = np.mean(c3)

#Filter spike train first, then compute costs



#Save the output matrix 
outfile = './data/output/param_w_N_%d_nsims_%d_c_%f_default_costfuncs.npz'%(N, nsims, c)
np.savez(outfile, costs2 = costs2, costs3 = costs3, costs2_filters = costs2_filtered, costs3_filtered = costs3_filtered)

