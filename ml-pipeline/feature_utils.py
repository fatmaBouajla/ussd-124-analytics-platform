"""

Fonctions et constantes partagees par tous les scripts qui entrainent ou

utilisent les modeles de prevision (RF/XGBoost/Holt-Winters, SHAP, confiance).



But : un seul endroit a modifier si la definition des features change,

au lieu de la dupliquer dans 6 scripts differents.

"""



SERIES = ["ca_reel", "nb_transactions", "taux_echec"]

TEMPOREL = ["heure_du_jour", "jour_semaine", "est_weekend"]



# Variante de features (complete/reduite) retenue empiriquement par cible,

# d'apres les resultats reels de train_forecast_models.py (aout 2026) :

#   ca_reel         -> reduit  (MAE 447.25 vs 514.56 complet)

#   nb_transactions -> reduit  (MAE 141.30 vs 149.01 complet)

#   taux_echec      -> complet (MAE 0.0148 vs 0.0154 reduit)

VARIANTE_GAGNANTE = {

    "ca_reel": "reduit",

    "nb_transactions": "reduit",

    "taux_echec": "complet",

}





def features_propres(serie):

    """Features basees uniquement sur l'historique de la serie elle-meme."""

    return [f"{serie}_moins_1h", f"{serie}_moins_2h", f"{serie}_moins_24h",
            f"{serie}_moyenne_3h", f"{serie}_moyenne_6h"]


def features_croisees_completes(serie):
    """Toutes les features propres des 2 autres series (variante 'complete')."""
    autres = [s for s in SERIES if s != serie]
    resultat = []
    for s in autres:
        resultat += features_propres(s)
    return resultat


def features_croisees_reduites(serie):
    """Version allegee : seulement moins_1h et moyenne_3h des autres series."""
    autres = [s for s in SERIES if s != serie]
    resultat = []
    for s in autres:
        resultat += [f"{s}_moins_1h", f"{s}_moyenne_3h"]
    return resultat


def colonnes_finales(serie):
    """
    Liste complete des colonnes features pour une cible donnee, selon la
    variante gagnante determinee empiriquement. C'est LA fonction a utiliser
    partout pour entrainer ou predire - ne jamais reconstruire cette liste
    a la main dans un script individuel.
    """
    croisees = (features_croisees_completes(serie)
                if VARIANTE_GAGNANTE[serie] == "complet"
                else features_croisees_reduites(serie))
    return features_propres(serie) + croisees + TEMPOREL
