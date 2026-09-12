


SERIES = ["ca_reel", "nb_transactions", "taux_echec"]

TEMPOREL = ["heure_du_jour", "jour_semaine", "est_weekend"]




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
    """variante 'complete'."""
    autres = [s for s in SERIES if s != serie]
    resultat = []
    for s in autres:
        resultat += features_propres(s)
    return resultat


def features_croisees_reduites(serie):
    """Version allegee """
    autres = [s for s in SERIES if s != serie]
    resultat = []
    for s in autres:
        resultat += [f"{s}_moins_1h", f"{s}_moyenne_3h"]
    return resultat


def colonnes_finales(serie):
 
    croisees = (features_croisees_completes(serie)
                if VARIANTE_GAGNANTE[serie] == "complet"
                else features_croisees_reduites(serie))
    return features_propres(serie) + croisees + TEMPOREL
