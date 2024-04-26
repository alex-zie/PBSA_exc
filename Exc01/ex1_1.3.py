import numpy as np
import matplotlib.pyplot as plt

n = 9  
dt = 1
dx = 1
X = range(n)
T = range(11)  
U = np.zeros((n, len(T)))  # Lösungsvektor

# Diskreter 1D Laplace
L = np.diag(np.ones(n-1), -1) + np.diag(np.ones(n-1), 1) - 2*np.diag(np.ones(n))

alpha = 1  # Diffusion coefficient

# Anfangsbedingung
U[4, 0] = 8

I = np.eye(n)

# impliziter Euler
for i in range(1, len(T)):
    A = (I - dt * alpha * L / dx**2)
    U[:, i] = np.linalg.solve(A, U[:, i-1])

for t in [0, 1, 4, 10]: 
    plt.plot(X, U[:, t], label=f"t={t}")
plt.legend()
plt.title("Lösung der Wärme-PDE in 1D")
plt.xlabel("x")
plt.ylabel("Temperatur")
plt.show()