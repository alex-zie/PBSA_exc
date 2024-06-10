import numpy as np
import scipy
import scipy.linalg 

def conjugate_gradients(A, b, x0=None, tol=1e-3, max_iter=25):
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
    b = b.flatten() # damit inner() bei Spaltenvektoren gescheit funktioniert

    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0
    r = b - A@x
    d = r.copy()
    
    for i in range(max_iter):
        r_old = r.copy()
        Ad = A@d
        alpha = np.inner(r, r) / np.inner(d, Ad) # inner, damit ich nicht transponieren muss
        x += alpha * d
        r -= alpha * Ad
        
        if np.linalg.norm(r) < tol:
            break
        
        beta = (np.inner(r, r) / np.inner(r_old, r_old))
        d = r + beta*d

    print(f"Terminated after {i+1} iterations.")
    return x
