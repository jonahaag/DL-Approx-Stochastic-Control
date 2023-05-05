import numpy as np
import matplotlib.pyplot as plt

"------Configuration---------"
S_0 = 10
mu = 10
sigma = 2
N = 100 #Timesteps
S = np.zeros(N)
S[0] = S_0
d = np.random.normal(mu, sigma, N)
for i in range(N-1):
    S[i+1] = state_update(S[i], 1, d[i])
    
def state_update(s, i, d):
    return s+i-d

plt.plot(s)