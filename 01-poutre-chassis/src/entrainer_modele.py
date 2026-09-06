"""
Entraîne et compare deux modèles pour prédire flèche et contrainte :
1. Régression linéaire sur log(entrées) et log(sorties) -- exploite le fait que
   la physique du problème est une loi de puissance (donc linéaire en log-log)
2. Random Forest sur les mêmes données log-log, pour comparaison
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

DOSSIER_SCRIPT = Path(__file__).resolve().parent
DOSSIER_DATA = DOSSIER_SCRIPT.parent / "data"
DOSSIER_MODELES = DOSSIER_SCRIPT.parent / "modeles"
DOSSIER_MODELES.mkdir(exist_ok=True)

df = pd.read_csv(DOSSIER_DATA / "dataset_poutre.csv")

# Entrées ET sorties en log cette fois
X_log = np.log(df[["L", "b", "h", "F"]])
y_log = np.log(df[["fleche_mm", "sigma_MPa"]])

X_train, X_test, y_train_log, y_test_log = train_test_split(
    X_log, y_log, test_size=0.2, random_state=42
)

def evaluer(nom, modele):
    modele.fit(X_train, y_train_log)
    y_pred = np.exp(modele.predict(X_test))
    y_test = np.exp(y_test_log)
    r2_f = r2_score(y_test["fleche_mm"], y_pred[:, 0])
    r2_s = r2_score(y_test["sigma_MPa"], y_pred[:, 1])
    mae_f = mean_absolute_error(y_test["fleche_mm"], y_pred[:, 0])
    mae_s = mean_absolute_error(y_test["sigma_MPa"], y_pred[:, 1])
    print(f"\n=== {nom} ===")
    print(f"Flèche     : R² = {r2_f:.4f}  |  Erreur moyenne = {mae_f:.4f} mm")
    print(f"Contrainte : R² = {r2_s:.4f}  |  Erreur moyenne = {mae_s:.4f} MPa")
    return modele

lin = evaluer("Régression linéaire (log-log)", LinearRegression())
rf = evaluer("Random Forest (log-log)", RandomForestRegressor(
    n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1
))

# On garde le meilleur des deux (probablement la régression linéaire ici)
joblib.dump(lin, DOSSIER_MODELES / "modele_poutre.pkl")
print(f"\nModèle linéaire sauvegardé dans {DOSSIER_MODELES / 'modele_poutre.pkl'}")
print("Note : ce modèle prend log(L,b,h,F) en entrée et prédit log(flèche, contrainte).")