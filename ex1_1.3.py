import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

n = 9
dt = 1
X = range(n)
T = range(11)
U = np.zeros((n,len(T))) # Lösungsvektor

# Diskreter 1D Laplace
L = -np.diag(np.ones(n-1), -1) - np.diag(np.ones(n-1), 1) + 2*np.diag(np.ones(n))

alpha = 100

#Anfangsbedingung
U[4,0] = 8

for i in range(1, len(T)):
    U[:,i] = U[:,i-1] + dt*alpha*L@U[:,i-1]

plt.plot(X, U[:,0], label="t=0")
plt.plot(X, U[:,1], label="t=1")
#plt.plot(X, U[:,4], label="t=4")
#plt.plot(X, U[:,10], label="t=10")
plt.legend()
plt.show()