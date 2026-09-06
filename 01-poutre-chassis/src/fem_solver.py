"""
Solveur éléments finis pour une poutre en flexion (théorie d'Euler-Bernoulli).
Chaque noeud a 2 degrés de liberté : déplacement vertical (v) et rotation (theta).
"""
import numpy as np
from analytique import moment_inertie_rectangle


def matrice_rigidite_element(E, I, Le):
    """Matrice de rigidité d'un élément de poutre (4x4).
    Ordre des DDL : [v1, theta1, v2, theta2]
    """
    k = (E * I / Le**3) * np.array([
        [12,      6*Le,    -12,     6*Le],
        [6*Le,    4*Le**2, -6*Le,   2*Le**2],
        [-12,    -6*Le,     12,    -6*Le],
        [6*Le,    2*Le**2, -6*Le,   4*Le**2]
    ])
    return k


def assembler_rigidite_globale(n_elements, E, I, Le):
    """Assemble la matrice de rigidité globale à partir des matrices élémentaires."""
    n_noeuds = n_elements + 1
    n_ddl = 2 * n_noeuds
    K = np.zeros((n_ddl, n_ddl))
    ke = matrice_rigidite_element(E, I, Le)

    for e in range(n_elements):
        idx = [2*e, 2*e + 1, 2*e + 2, 2*e + 3]
        for i in range(4):
            for j in range(4):
                K[idx[i], idx[j]] += ke[i, j]
    return K


def resoudre_poutre(L, b, h, F, E, n_elements=10):
    """Résout le problème : poutre encastrée-libre avec charge F en bout."""
    I = moment_inertie_rectangle(b, h)
    Le = L / n_elements
    n_noeuds = n_elements + 1
    n_ddl = 2 * n_noeuds

    K = assembler_rigidite_globale(n_elements, E, I, Le)

    # Vecteur des forces : charge F appliquée au dernier noeud (DDL vertical)
    Fvec = np.zeros(n_ddl)
    Fvec[-2] = -F

    # Conditions aux limites : noeud 0 encastré → on retire ses 2 DDL (v=0, theta=0)
    ddl_libres = list(range(2, n_ddl))

    K_reduit = K[np.ix_(ddl_libres, ddl_libres)]
    F_reduit = Fvec[ddl_libres]

    U_reduit = np.linalg.solve(K_reduit, F_reduit)

    U = np.zeros(n_ddl)
    U[ddl_libres] = U_reduit

    fleche_max = abs(U[-2])  # déplacement vertical du dernier noeud

    # Contrainte max à l'encastrement (moment fléchissant max = F * L pour ce cas)
    M_max = F * L
    sigma_max = (M_max * h / 2) / I

    return fleche_max, sigma_max


if __name__ == "__main__":
    E = 210e9
    L = 1.0
    b = 0.05
    h = 0.05
    F = 1000

    fleche, sigma = resoudre_poutre(L, b, h, F, E, n_elements=10)
    print(f"[FEM] Flèche max = {fleche*1000:.3f} mm")
    print(f"[FEM] Contrainte max = {sigma/1e6:.2f} MPa")
    