import pandas as pd

import shap

from sklearn.ensemble import RandomForestRegressor



from feature_utils import SERIES, colonnes_finales



FICHIER_TRAIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_train.csv"

FICHIER_TEST = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_test.csv"





def main():

    train = pd.read_csv(FICHIER_TRAIN, parse_dates=["datetime"])

    test = pd.read_csv(FICHIER_TEST, parse_dates=["datetime"])

    donnees = pd.concat([train, test], ignore_index=True)



    for serie in SERIES:

        cible = f"{serie}_cible"

        colonnes = colonnes_finales(serie)



        modele = RandomForestRegressor(n_estimators=200, random_state=42)

        modele.fit(donnees[colonnes], donnees[cible])



        explainer = shap.TreeExplainer(modele)

        valeurs_shap = explainer.shap_values(donnees[colonnes])



        importance_moyenne = pd.Series(

            abs(valeurs_shap).mean(axis=0), index=colonnes

        ).sort_values(ascending=False)



        print(f"\n=== SHAP - importance moyenne des variables ({serie}) ===")
        print(importance_moyenne.head(8))

        fichier_sortie = f"/home/fatma/elk-ussd-orange/ml-pipeline/data/shap_{serie}.csv"
        importance_moyenne.to_csv(fichier_sortie, header=["importance_shap"])
        print(f"Fichier ecrit : {fichier_sortie}")


if __name__ == "__main__":
    main()
