import numpy as np
import scipy
import scipy.linalg 

def preconditioned_cg(A, b, x0=None, tol=1e-3, max_iter=25, omega=0):
    """
    Implementiert die Methode der konjugierten Gradienten für die Lösung von Ax = b.

    Parameter:
        A (scipy.sparse.*)
        b (numpy.array): Rechte Seite des linearen Gleichungssystems.
        x0 (numpy.array): Startvektor für die Iteration.
        tol (float): Toleranz für das Konvergenzkriterium.
        max_iter (int): Maximale Anzahl von Iterationen.

    Rückgabe:
        x (numpy.array): Lösung des Gleichungssystems.
    """
    if omega < 0 or omega > 1:
        print("omega muss zwischen 0 und 1 liegen!")
        return
    
    # preconditioning
    D = np.diag(np.diag(A.toarray())) # Diagonalteil
    L = np.tril(A.toarray(), -1) # unterer Dreiecksteil
    if (L + D + L.T != A).all():
        print("Matrix A ist nicht symmetrisch!")
        return
    
    if omega == 0:
        M = D
    else:
        Y = np.linalg.solve(D, (1/omega*D+L.T)) # vermeide Berechnug der Inversen von D
        M = (omega/(2-omega)*(1/omega*D+L))@Y

    b = b.flatten() # damit inner() bei Spaltenvektoren gescheit funktioniert

    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0
    r = b - A@x
    z = np.linalg.solve(M, r)
    p = z.copy()
    
    for i in range(max_iter):
        r_old = r.copy()
        z_old = z.copy()
        Ap = A@p
        alpha = np.inner(r, z) / np.inner(p, Ap) # inner, damit ich nicht transponieren muss
        x += alpha * p
        r -= alpha * Ap
        
        if np.linalg.norm(r) < tol:
            break
        
        z = np.linalg.solve(M, r)
        beta = (np.inner(z, r) / np.inner(z_old, r_old))
        p = z + beta*p

    print(f"Terminated after {i+1} iterations.")
    return x