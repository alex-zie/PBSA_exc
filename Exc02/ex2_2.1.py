import numpy as np

# 2.1a)
def inertia_tensor(a, b, c, rho):
    J = np.zeros((3, 3))
    # for i in range(3):
    #     for j in range(3):
    #         J[i,j] = 0
    mass = rho * (a * b * c)
    J[0,0] = 1/12 * mass * (b ** 2 + c ** 2)
    J[1,1] = 1/12 * mass * (a ** 2 + c ** 2)
    J[2,2] = 1/12 * mass * (a ** 2 + b ** 2)
    return J

J = inertia_tensor(1, 1, 1, 1)
print(J)

# 2.1b)
rot_ang = np.pi/4
R = np.array([[np.cos(rot_ang), -np.sin(rot_ang), 0], [np.sin(rot_ang), np.cos(rot_ang), 0], [0, 0, 1]])

J_rot = R * J * R.T # ist das richtig?
print(J_rot)

# 2.1c)
# L = J * omega
# Winkelgeschwindigkeit (omega): wird größer da der Drehimpuls gleich bleiben muss und der Trägheitstensor kleiner wird
# Drehimpuls (L): bleibt gleich da keine exterene Energie dazu kommt
# Trägheitstensor (J): wird kleiner da die Masse näher an der Drehachse ist
