"""
Formules analytiques de référence (théorie d'Euler-Bernoulli)
pour une poutre encastrée-libre avec charge ponctuelle en bout.
Sert à valider notre solveur FEM.
"""

def moment_inertie_rectangle(b, h):
    """Moment quadratique d'une section rectangulaire (m^4)."""
    return (b * h**3) / 12

def fleche_max_analytique(F, L, E, I):
    """Flèche maximale en bout de poutre (m)."""
    return (F * L**3) / (3 * E * I)

def contrainte_max_analytique(F, L, h, I):
    """Contrainte de flexion maximale à l'encastrement (Pa)."""
    M = F * L  # moment de flexion maximal
    return (M * h / 2) / I

if __name__ == "__main__":
    # Exemple de test
    E = 210e9    # Module de Young de l'acier (Pa)
    L = 1.0      # Longueur (m)
    b = 0.05     # Largeur (m)
    h = 0.05     # Hauteur (m)
    F = 1000     # Force (N)

    I = moment_inertie_rectangle(b, h)
    delta = fleche_max_analytique(F, L, E, I)
    sigma = contrainte_max_analytique(F, L, h, I)

    print(f"Moment quadratique I = {I:.3e} m^4")
    print(f"Flèche max = {delta*1000:.3f} mm")
    print(f"Contrainte max = {sigma/1e6:.2f} MPa")