import numpy as np
import matplotlib.pyplot as plt

# dt = np.pi / 20, t in [0, 12 * π)
t = np.linspace(0, 12 * np.pi, 240, endpoint=False)
theta0 = np.pi / 50
omega0 = 0.0
g = 5.0
l = 5.0
thetaref = theta0 * np.cos(np.sqrt(g/l)*t)


def explicitEuler(t, theta0, omega0, g, l):
    theta = [theta0]
    omega = [omega0]
    t_last = t[0]
    for k, t_cur in enumerate(t):
        if k == 0:
            continue
        dt, t_last = t_cur - t_last, t_cur
        omega.append(...) # selber füllen
        theta.append(...) # selber füllen
    return np.asarray(theta)


plt.plot(t, thetaref, label="Small Angle")
plt.plot(t, explicitEuler(t, theta0, omega0, g, l), label="Explicit Euler")
plt.legend()
plt.show()
