import pandas as pd

# Etape 1:
# Lancer les régressions linéaires pour chaque modèle de voiture s'il y a au moins 30 observations
# Chaque modèle de voiture est attribué à une équation de régression linéaire
# y = ratio_vr
# x = ['age_months', 'kilometrage']


# Etape 2:
# Sur les modèles qui n'ont pas au moins 30 observations:
# On utilise K-means pour attribuer le modèle de voiture à un des modèles de l'étape 1 


