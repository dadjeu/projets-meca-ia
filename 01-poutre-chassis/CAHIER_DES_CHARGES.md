# Cahier des charges — Projet 1 : Prédiction de la déformation et de la contrainte d'une poutre en porte-à-faux par modèle de substitution (Machine Learning)

## 1. Contexte
Le dimensionnement mécanique de pièces structurelles (ex: longerons de châssis 
automobile) repose classiquement sur des simulations par éléments finis (FEM), 
coûteuses en temps de calcul lorsqu'il faut explorer de nombreuses variantes 
géométriques (étude paramétrique, optimisation, conception itérative).

L'objectif de ce projet est de démontrer la faisabilité d'un modèle de substitution 
(surrogate model) basé sur le Machine Learning, capable de prédire quasi 
instantanément le comportement mécanique d'une pièce, en remplacement d'une 
simulation FEM classique.

## 2. Cas d'étude
Poutre en porte-à-faux (encastrée à une extrémité, libre à l'autre), section 
rectangulaire constante, soumise à une charge ponctuelle verticale appliquée 
à l'extrémité libre.

## 3. Objectifs
- Développer un solveur éléments finis (FEM) pour ce cas, validé analytiquement
- Générer un jeu de données de simulations en faisant varier la géométrie
- Entraîner un modèle ML prédisant :
  - la flèche maximale (déformation en bout de poutre)
  - la contrainte maximale (risque de rupture)
- Comparer les performances (précision, temps de calcul) du modèle ML face au 
  solveur FEM classique
- Développer une interface web permettant de tester le modèle interactivement
- Documenter l'ensemble dans un rapport scientifique structuré

## 4. Paramètres d'entrée (variables du problème)
| Paramètre | Symbole | Plage de variation |
|---|---|---|
| Longueur de la poutre | L | 0.5 – 3 m |
| Largeur de la section | b | 0.02 – 0.15 m |
| Hauteur de la section | h | 0.02 – 0.15 m |
| Charge appliquée | F | 100 – 5000 N |
| Matériau | E, ρ | Acier (E = 210 GPa) fixé pour ce projet |

## 5. Sorties attendues (variables prédites)
- Flèche maximale δ_max (mm)
- Contrainte maximale σ_max (MPa)

## 6. Contraintes techniques
- Langage : Python
- Outils gratuits et open-source uniquement
- Solveur FEM développé "from scratch" (pédagogique, pas de logiciel commercial)
- Application finale déployée gratuitement sur le web

## 7. Livrables
1. Solveur FEM (code Python documenté)
2. Jeu de données de simulations (CSV)
3. Modèle(s) ML entraîné(s) et comparés
4. Application web interactive (Streamlit)
5. Rapport scientifique (LaTeX → PDF)
6. Dépôt GitHub complet et documenté
7. Publication LinkedIn

## 8. Critères de réussite
- Le solveur FEM doit correspondre à la théorie d'Euler-Bernoulli à moins de 5% d'erreur
- Le modèle ML doit atteindre un R² > 0.95 sur le jeu de test
- L'application web doit être accessible publiquement via une URL
- Le rapport doit suivre une structure scientifique standard (IMRaD)

## 9. Planning
| Jour | Tâche |
|---|---|
| 1 | Cahier des charges (ce document) |
| 2 | Solveur FEM |
| 3 | Génération de données + premier modèle ML |
| 4 | Validation + optimisation |
| 5 | Application web |
| 6 | Rapport LaTeX |
| 7 | Déploiement + publication |