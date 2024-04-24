import numpy as np
import matplotlib.pyplot as plt

# dt = np.pi / 20, t in [0, 12 * π)
t = np.linspace(0, 12 * np.pi, 240, endpoint=False)
theta0 = np.pi / 50
omega0 = 0.0
g = 5.0
l = 5.0
thetaref = theta0 * np.cos(np.sqrt(g/l)*t)

f = lambda t, y1, y2: np.array([y2, -g/l * np.sin(y1)])

def explicitEuler(t, theta0, omega0, g, l):
    theta = [theta0]
    omega = [omega0]
    y = np.array([theta, omega])
    t_last = t[0]
    for k, t_cur in enumerate(t):
        if k == 0:
            continue
        dt, t_last = t_cur - t_last, t_cur
        y = [theta[k-1], omega[k-1]] + dt*f(t_cur, theta[k-1], omega[k-1])
        theta.append(y[0]) # selber füllen
        omega.append(y[1]) # selber füllen
    return np.asarray(theta)

t0 = 0
T = 12 * np.pi
dt = 10e-2
t = range(0, T, dt)

f = lambda t, y: np.array([y[1], -g/l * np.sin(y[0])])

def implicitEuler(t, theta0, omega0, g, l):
    y = np.zeros((2, len(t))) # fist axis dims, second axis time
    y[0,0] = theta0
    y[1,0] = omega0

    for i in enumerate(t):
        if i == 0:
            continue

        y[i] = y[i-1] + f(t[i], y[i])


    return


plt.plot(t, thetaref, label="Small Angle")
plt.plot(t, explicitEuler(t, theta0, omega0, g, l), label="Explicit Euler")
plt.plot(t, implicitEuler(t, theta0, omega0, g, l), label="Implicit Euler")
plt.legend()
plt.show()
