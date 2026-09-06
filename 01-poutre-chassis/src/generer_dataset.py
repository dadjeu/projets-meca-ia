"""
Génère un jeu de données de simulations FEM en faisant varier
la géométrie de la poutre. Tirage log-uniforme pour les paramètres
géométriques (car leur effet est en loi de puissance), et filtrage
des cas où la déformation dépasse la validité de la théorie linéaire.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from fem_solver import resoudre_poutre

DOSSIER_SCRIPT = Path(__file__).resolve().parent
DOSSIER_DATA = DOSSIER_SCRIPT.parent / "data"
DOSSIER_DATA.mkdir(exist_ok=True)

np.random.seed(42)

E = 210e9
N_SIMULATIONS = 3000  # on augmente le volume de données

def tirage_log_uniforme(mini, maxi, n):
    """Tire n valeurs uniformément réparties en échelle logarithmique."""
    return np.exp(np.random.uniform(np.log(mini), np.log(maxi), n))

L_vals = tirage_log_uniforme(0.5, 3.0, N_SIMULATIONS)
b_vals = tirage_log_uniforme(0.02, 0.15, N_SIMULATIONS)
h_vals = tirage_log_uniforme(0.02, 0.15, N_SIMULATIONS)
F_vals = tirage_log_uniforme(100, 5000, N_SIMULATIONS)

resultats = []
n_rejetes = 0

for i in range(N_SIMULATIONS):
    L, b, h, F = L_vals[i], b_vals[i], h_vals[i], F_vals[i]
    fleche, sigma = resoudre_poutre(L, b, h, F, E, n_elements=10)

    # Validité de la théorie linéaire : flèche < 10% de la longueur
    if fleche > 0.1 * L:
        n_rejetes += 1
        continue

    resultats.append({
        "L": L, "b": b, "h": h, "F": F,
        "fleche_mm": fleche * 1000,
        "sigma_MPa": sigma / 1e6
    })

df = pd.DataFrame(resultats)
df.to_csv(DOSSIER_DATA / "dataset_poutre.csv", index=False)
print(f"{len(df)} simulations valides conservées ({n_rejetes} rejetées : hors domaine de validité linéaire)")
print(f"Dataset sauvegardé dans {DOSSIER_DATA / 'dataset_poutre.csv'}")
print(df.describe())