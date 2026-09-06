"""
Application web interactive : prédiction de la déformation et de la contrainte
d'une poutre en porte-à-faux, via modèle ML (surrogate model), avec comparaison
au solveur FEM et optimisation de la géométrie.
"""
import streamlit as st
import numpy as np
import joblib
from pathlib import Path
from fem_solver import resoudre_poutre
from scipy.optimize import minimize

DOSSIER_SCRIPT = Path(__file__).resolve().parent
DOSSIER_MODELES = DOSSIER_SCRIPT.parent / "modeles"
modele = joblib.load(DOSSIER_MODELES / "modele_poutre.pkl")

E = 210e9  # Acier
LIMITE_ELASTIQUE_ACIER = 250  # MPa

st.set_page_config(page_title="Poutre - Surrogate Model", layout="centered")
st.title("Prédiction du comportement d'une poutre en porte-à-faux")
st.markdown(
    "Modèle de substitution (surrogate model) entraîné sur simulations FEM, "
    "pour prédire instantanément la flèche et la contrainte d'une poutre en acier "
    "encastrée-libre soumise à une charge en bout."
)

st.header("1. Paramètres de la poutre")
col1, col2 = st.columns(2)
with col1:
    L = st.slider("Longueur L (m)", 0.5, 3.0, 1.5, 0.05)
    b = st.slider("Largeur b (mm)", 20.0, 150.0, 60.0, 1.0) / 1000
with col2:
    F = st.slider("Charge F (N)", 100, 5000, 2000, 50)
    h = st.slider("Hauteur h (mm)", 20.0, 150.0, 80.0, 1.0) / 1000

# Prédiction ML
entree_log = np.log([[L, b, h, F]])
fleche_ml, sigma_ml = np.exp(modele.predict(entree_log)[0])

# Calcul FEM réel pour comparaison
fleche_fem, sigma_fem_pa = resoudre_poutre(L, b, h, F, E, n_elements=10)
fleche_fem *= 1000
sigma_fem = sigma_fem_pa / 1e6

st.header("2. Résultats : modèle ML vs calcul FEM")
col1, col2 = st.columns(2)
with col1:
    st.metric("Flèche prédite (ML)", f"{fleche_ml:.3f} mm")
    st.metric("Flèche calculée (FEM)", f"{fleche_fem:.3f} mm")
with col2:
    st.metric("Contrainte prédite (ML)", f"{sigma_ml:.2f} MPa")
    st.metric("Contrainte calculée (FEM)", f"{sigma_fem:.2f} MPa")

marge_securite = LIMITE_ELASTIQUE_ACIER / sigma_fem if sigma_fem > 0 else float("inf")
if sigma_fem > LIMITE_ELASTIQUE_ACIER:
    st.error(f"⚠️ Contrainte supérieure à la limite élastique de l'acier ({LIMITE_ELASTIQUE_ACIER} MPa) — risque de déformation permanente.")
else:
    st.success(f"Coefficient de sécurité actuel : {marge_securite:.2f}")

st.divider()

st.header("3. Optimisation : trouver la géométrie la plus légère")
st.markdown("Pour une longueur et une charge données, trouve `b` et `h` minimisant le volume de matière sous contraintes.")

col1, col2 = st.columns(2)
with col1:
    fleche_max_admissible = st.number_input("Flèche maximale admissible (mm)", 0.1, 50.0, 5.0, 0.1)
with col2:
    coeff_securite = st.number_input("Coefficient de sécurité souhaité", 1.0, 5.0, 1.5, 0.1)

if st.button("Lancer l'optimisation"):
    sigma_admissible = LIMITE_ELASTIQUE_ACIER / coeff_securite

    def predire(params):
        b_opt, h_opt = params
        entree = np.log([[L, b_opt, h_opt, F]])
        return np.exp(modele.predict(entree)[0])

    def volume(params):
        b_opt, h_opt = params
        return L * b_opt * h_opt

    def contraintes(params):
        fleche, sigma = predire(params)
        return [fleche_max_admissible - fleche, sigma_admissible - sigma]

    resultat = minimize(
        volume, x0=[0.06, 0.08],
        bounds=[(0.02, 0.15), (0.02, 0.15)],
        constraints=[{"type": "ineq", "fun": lambda p: np.array(contraintes(p))}],
        method="SLSQP"
    )

    b_opt, h_opt = resultat.x
    fleche_opt, sigma_opt = predire([b_opt, h_opt])

    st.success("Optimisation terminée")
    col1, col2, col3 = st.columns(3)
    col1.metric("b optimal", f"{b_opt*1000:.1f} mm")
    col2.metric("h optimal", f"{h_opt*1000:.1f} mm")
    col3.metric("Volume", f"{resultat.fun*1e6:.1f} cm³")
    st.write(f"Flèche obtenue : {fleche_opt:.3f} mm (limite : {fleche_max_admissible} mm)")
    st.write(f"Contrainte obtenue : {sigma_opt:.2f} MPa (limite : {sigma_admissible:.1f} MPa)")

st.divider()
st.caption(
    "Projet réalisé dans le cadre d'un portfolio de modélisation mécanique + IA. "
    "Solveur FEM développé from scratch (théorie d'Euler-Bernoulli), "
    "modèle de substitution entraîné en log-log (régression linéaire, R²=1.0 sur ce cas linéaire)."
)