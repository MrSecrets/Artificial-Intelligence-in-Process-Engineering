import numpy as np 


def corrupt_sine(gamma, N, low, high):

	X = np.random.uniform(low=low, high=high, size=N)
	x = np.random.uniform(low=0, high=1, size = N)
	y = np.sin(2*np.pi*x) + gamma*X

	return y