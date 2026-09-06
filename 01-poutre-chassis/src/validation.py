"""
Validation visuelle du modèle : 
1. Comparaison prédictions vs valeurs réelles (graphiques)
2. Comparaison du temps de calcul FEM vs modèle ML
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path
import joblib
from fem_solver import resoudre_poutre

DOSSIER_SCRIPT = Path(__file__).resolve().parent
DOSSIER_DATA = DOSSIER_SCRIPT.parent / "data"
DOSSIER_MODELES = DOSSIER_SCRIPT.parent / "modeles"
DOSSIER_FIGURES = DOSSIER_SCRIPT.parent / "figures"
DOSSIER_FIGURES.mkdir(exist_ok=True)

# 1. Charger modèle et données
modele = joblib.load(DOSSIER_MODELES / "modele_poutre.pkl")
df = pd.read_csv(DOSSIER_DATA / "dataset_poutre.csv")

X_log = np.log(df[["L", "b", "h", "F"]])
y_log = np.log(df[["fleche_mm", "sigma_MPa"]])
y_pred_log = modele.predict(X_log)
y_pred = np.exp(y_pred_log)
y_reel = np.exp(y_log)

# 2. Graphique : prédiction vs réalité
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_reel["fleche_mm"], y_pred[:, 0], alpha=0.3, s=10)
lim = [y_reel["fleche_mm"].min(), y_reel["fleche_mm"].max()]
axes[0].plot(lim, lim, 'r--', label="Prédiction parfaite")
axes[0].set_xlabel("Flèche réelle (FEM, mm)")
axes[0].set_ylabel("Flèche prédite (ML, mm)")
axes[0].set_title("Flèche : ML vs FEM")
axes[0].legend()

axes[1].scatter(y_reel["sigma_MPa"], y_pred[:, 1], alpha=0.3, s=10, color="orange")
lim = [y_reel["sigma_MPa"].min(), y_reel["sigma_MPa"].max()]
axes[1].plot(lim, lim, 'r--', label="Prédiction parfaite")
axes[1].set_xlabel("Contrainte réelle (FEM, MPa)")
axes[1].set_ylabel("Contrainte prédite (ML, MPa)")
axes[1].set_title("Contrainte : ML vs FEM")
axes[1].legend()

plt.tight_layout()
plt.savefig(DOSSIER_FIGURES / "validation_ml_vs_fem.png", dpi=150)
print(f"Graphique sauvegardé dans {DOSSIER_FIGURES / 'validation_ml_vs_fem.png'}")

# 3. Comparaison du temps de calcul
E = 210e9
L, b, h, F = 1.5, 0.06, 0.08, 2000

n_repets = 200

# Temps FEM
t0 = time.perf_counter()
for _ in range(n_repets):
    resoudre_poutre(L, b, h, F, E, n_elements=10)
t_fem = (time.perf_counter() - t0) / n_repets

# Temps ML
entree_log = np.log([[L, b, h, F]])
t0 = time.perf_counter()
for _ in range(n_repets):
    modele.predict(entree_log)
t_ml = (time.perf_counter() - t0) / n_repets

print(f"\n=== Comparaison des temps de calcul (moyenne sur {n_repets} appels) ===")
print(f"FEM : {t_fem*1000:.4f} ms par appel")
print(f"ML  : {t_ml*1000:.4f} ms par appel")
print(f"Accélération : x{t_fem/t_ml:.1f}")

plt.show()