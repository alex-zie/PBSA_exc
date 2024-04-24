import numpy as np
import matplotlib.pyplot as plt

# dt = np.pi / 20, t in [0, 12 * π)
t1 = np.linspace(0, 12 * np.pi, 240, endpoint=False)
theta0 = np.pi / 50
omega0 = 0.0
g = 5.0
l = 5.0
thetaref = theta0 * np.cos(np.sqrt(g/l)*t1)

f1 = lambda t, y1, y2: np.array([y2, -g/l * np.sin(y1)])

def explicitEuler(t, theta0, omega0, g, l):
    theta = [theta0]
    omega = [omega0]
    y = np.array([theta, omega])
    t_last = t[0]
    for k, t_cur in enumerate(t):
        if k == 0:
            continue
        dt, t_last = t_cur - t_last, t_cur
        y = [theta[k-1], omega[k-1]] + dt*f1(t_cur, theta[k-1], omega[k-1])
        theta.append(y[0]) # selber füllen
        omega.append(y[1]) # selber füllen
    return np.asarray(theta)

t0 = 0 # start
T = 12 * np.pi # stop
dt = np.pi / 20 # time step
t2 = np.arange(t0, T, dt)

f2 = lambda t, y: np.array([y[1], -g/l * np.sin(y[0])])

def implicitEuler(t, theta0, omega0, g, l):
    y = np.zeros((2, len(t))) # fist axis dims, second axis time
    y[0,0] = theta0
    y[1,0] = omega0

    for i, _ in enumerate(t):
        if i == 0:
            continue
        y[1,i] = (l / (dt * g) * y[1,i-1] - np.sin(y[0,i-1])) / (l / (dt * g) + dt * np.cos(y[0,i-1]))
        y[0,i] = y[0,i-1] + dt * y[1,i]
    return y[0]

plt.plot(t1, thetaref, label="Small Angle")
plt.plot(t1, explicitEuler(t1, theta0, omega0, g, l), label="Explicit Euler")
plt.plot(t2, implicitEuler(t2, theta0, omega0, g, l), label="Implicit Euler")
plt.legend()
plt.show()
