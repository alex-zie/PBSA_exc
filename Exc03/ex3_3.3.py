import numpy as np
from scipy.sparse import csr_matrix

def conjugate_gradients(A, b, x0=None, tol=1e-10, max_iter=1000):
    """
    Implementiert die Methode der konjugierten Gradienten für die Lösung von Ax = b.

    Parameter:
        A (scipy.sparse.csr_matrix): Symmetrische, positiv definite Matrix.
        b (numpy.array): Rechte Seite des linearen Gleichungssystems.
        x0 (numpy.array): Startvektor für die Iteration.
        tol (float): Toleranz für das Konvergenzkriterium.
        max_iter (int): Maximale Anzahl von Iterationen.

    Rückgabe:
        x (numpy.array): Lösung des Gleichungssystems.
    """
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0
    r = b - A.dot(x)
    p = r.copy()
    rsold = np.inner(r, r)

    for i in range(max_iter):
        Ap = A.dot(p)
        alpha = rsold / np.inner(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.inner(r, r)
        
        if np.sqrt(rsnew) < tol:
            break
        
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x

# Beispielmatrix und Vektor b
n = 10
diagonals = np.arange(1, n + 1)
A = csr_matrix(np.diags(diagonals, 0))
b = np.random.rand(n)

# Lösen des Gleichungssystems
x = conjugate_gradients(A, b)

print("Lösung x:", x)
