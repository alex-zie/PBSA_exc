import numpy as np

x_0 = np.vstack(np.array((0.0, 0.0, 0.8)))
x_1 = np.vstack(np.array((1.0, 0.0, 0.5)))
x_2 = np.vstack(np.array((0.2, 0.2, 1.0)))
x_3 = np.vstack(np.array((1, -1, 1)))
x_mat = np.hstack((x_0, x_1, x_2, x_3))
# use pseudoinverse as not a square matrix
x_mat_inv = np.linalg.inv(np.transpose(x_mat) @ x_mat) @ np.transpose(x_mat)

xp_0 = np.vstack(np.array((5.2, 3.3, 7.5)))
xp_1 = np.vstack(np.array((7.0, 3.0, 7.0)))
xp_2 = np.vstack(np.array((7.199999999999999, 3.7, 9.5)))
xp_3 = np.vstack(np.array((6.0, 2.5, 5.5)))
xp_mat = np.hstack((xp_0, xp_1, xp_2, xp_3))
Ap = xp_mat @ x_mat_inv
print(Ap)

xpp_0 = np.vstack(np.array((1.6, 5.2, 1.5)))
xpp_1 = np.vstack(np.array((3.0, 5.0, 2.5)))
xpp_2 = np.vstack(np.array((3.0, 6.4, 2.7)))
xpp_3 = np.vstack(np.array((1.0, 6.0, -2.5)))
xpp_mat = np.hstack((xpp_0, xpp_1, xpp_2, xpp_3))
App = xpp_mat @ x_mat_inv
print(App)

xppp_0 = np.vstack(np.array((5.3, 4.7, 5.4)))
xppp_1 = np.vstack(np.array((8.0, 3.5, 7.5)))
xppp_2 = np.vstack(np.array((6.7, 6.5, 6.6)))
xppp_3 = np.vstack(np.array((5.5, 0.5, 9.0)))
xppp_mat = np.hstack((xppp_0, xppp_1, xppp_2, xppp_3))
Appp = xppp_mat @ x_mat_inv
print(Appp)

def polar_decomposition(A):
    '''Polar decomposition of matrix A'''
    AA_T = np.matmul(A, A.transpose())
    eigenvalues, eigenvectors = np.linalg.eig(AA_T)
    eigenvalues = np.abs(eigenvalues)
    S = np.diag(np.sqrt(eigenvalues))

    P = eigenvectors @ S
    P = P @ np.conjugate(eigenvectors.T)
    U = eigenvectors

    return U, P
