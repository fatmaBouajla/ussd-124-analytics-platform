import pandas as pd

import shap

from sklearn.ensemble import RandomForestRegressor



FICHIER_TRAIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_train.csv"

FICHIER_TEST = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_test.csv"

FICHIER_INCIDENT = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_incident.csv"

FICHIER_SORTIE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/shap_incident_detail.csv"



SERIES = ["ca_reel", "nb_transactions", "taux_echec"]

TEMPOREL = ["heure_du_jour", "jour_semaine", "est_weekend"]



VARIANTE_GAGNANTE = {

    "ca_reel": "reduit",

    "nb_transactions": "complet",

    "taux_echec": "reduit",

}





def features_propres(serie):

    return [f"{serie}_moins_1h", f"{serie}_moins_2h", f"{serie}_moins_24h",

            f"{serie}_moyenne_3h", f"{serie}_moyenne_6h"]





def features_croisees_completes(serie):

    autres = [s for s in SERIES if s != serie]

    resultat = []

    for s in autres:

        resultat += features_propres(s)

    return resultat


def features_croisees_reduites(serie):
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


def main():
    train = pd.read_csv(FICHIER_TRAIN)
    test = pd.read_csv(FICHIER_TEST)
    normal = pd.concat([train, test], ignore_index=True)

    incident = pd.read_csv(FICHIER_INCIDENT, parse_dates=["datetime"])

    serie = "ca_reel"
    cible = f"{serie}_cible"
    colonnes = colonnes_finales(serie)

    modele = RandomForestRegressor(n_estimators=200, random_state=42)
    modele.fit(normal[colonnes], normal[cible])

    explainer = shap.TreeExplainer(modele)
    valeurs_shap = explainer.shap_values(incident[colonnes])

    resultats = []
    for i, (_, ligne) in enumerate(incident.iterrows()):
        contributions = pd.Series(valeurs_shap[i], index=colonnes).sort_values()
        facteur_negatif = contributions.index[0]
        impact_negatif = contributions.iloc[0]
        facteur_positif = contributions.index[-1]
        impact_positif = contributions.iloc[-1]

        resultats.append({
            "datetime": ligne["datetime"],
            "ca_reel": ligne[cible],
            "facteur_qui_a_le_plus_baisse_la_prevision": facteur_negatif,
            "impact_dt": round(impact_negatif, 1),
            "facteur_qui_a_le_plus_augmente_la_prevision": facteur_positif,
            "impact_dt_positif": round(impact_positif, 1),
        })

    resultats_df = pd.DataFrame(resultats)
    resultats_df.to_csv(FICHIER_SORTIE, index=False)

    print("=== SHAP par creneau - explication de chaque prediction pendant l'incident ===")
    print(resultats_df.to_string(index=False))
    print(f"\nFichier ecrit : {FICHIER_SORTIE}")


if __name__ == "__main__":
    main()
