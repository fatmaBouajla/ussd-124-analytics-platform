import pandas as pd

import shap

from sklearn.ensemble import RandomForestRegressor



from feature_utils import colonnes_finales



FICHIER_TRAIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_train.csv"

FICHIER_TEST = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_test.csv"

FICHIER_INCIDENT = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_incident.csv"

FICHIER_SORTIE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/shap_incident_detail.csv"





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
