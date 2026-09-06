"""
Optimisation : trouve la géométrie (b, h) la plus légère (volume minimal)
pour une longueur L et une charge F données, en respectant :
- une flèche maximale admissible
- une contrainte maximale admissible (avec coefficient de sécurité)

Utilise le modèle ML (rapide) plutôt que le solveur FEM pour évaluer
des milliers de candidats en optimisation.
"""
import numpy as np
import joblib
from pathlib import Path
from scipy.optimize import minimize

DOSSIER_SCRIPT = Path(__file__).resolve().parent
DOSSIER_MODELES = DOSSIER_SCRIPT.parent / "modeles"

modele = joblib.load(DOSSIER_MODELES / "modele_poutre.pkl")

# Paramètres fixés du problème d'optimisation
L = 1.5          # Longueur de la poutre (m), fixée par le cahier des charges
F = 2000         # Charge appliquée (N)
FLECHE_MAX_ADMISSIBLE = 5.0    # mm
LIMITE_ELASTIQUE_ACIER = 250   # MPa
COEFF_SECURITE = 1.5
SIGMA_ADMISSIBLE = LIMITE_ELASTIQUE_ACIER / COEFF_SECURITE  # ≈ 167 MPa


def predire(b, h):
    """Utilise le modèle ML pour prédire flèche et contrainte."""
    entree_log = np.log([[L, b, h, F]])
    sortie_log = modele.predict(entree_log)[0]
    fleche, sigma = np.exp(sortie_log)
    return fleche, sigma


def volume(params):
    """Fonction objectif à minimiser : volume de matière (proportionnel au poids)."""
    b, h = params
    return L * b * h


def contraintes(params):
    """Retourne les marges de contraintes (doivent être >= 0 pour être respectées)."""
    b, h = params
    fleche, sigma = predire(b, h)
    marge_fleche = FLECHE_MAX_ADMISSIBLE - fleche
    marge_sigma = SIGMA_ADMISSIBLE - sigma
    return [marge_fleche, marge_sigma]


# Point de départ (estimation initiale raisonnable)
x0 = [0.06, 0.08]

# Bornes physiques (cohérentes avec le cahier des charges)
bornes = [(0.02, 0.15), (0.02, 0.15)]

contrainte_optim = {
    "type": "ineq",
    "fun": lambda params: np.array(contraintes(params))
}

resultat = minimize(
    volume, x0,
    bounds=bornes,
    constraints=[contrainte_optim],
    method="SLSQP"
)

b_opt, h_opt = resultat.x
fleche_opt, sigma_opt = predire(b_opt, h_opt)

print("=== Résultat de l'optimisation ===")
print(f"Largeur optimale b = {b_opt*1000:.2f} mm")
print(f"Hauteur optimale h = {h_opt*1000:.2f} mm")
print(f"Volume de matière = {resultat.fun*1e6:.2f} cm³")
print(f"Flèche obtenue = {fleche_opt:.3f} mm (limite : {FLECHE_MAX_ADMISSIBLE} mm)")
print(f"Contrainte obtenue = {sigma_opt:.2f} MPa (limite : {SIGMA_ADMISSIBLE:.1f} MPa)")