import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

from feature_utils import SERIES, colonnes_finales

FICHIER_TRAIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_train.csv"


def main():
    donnees = pd.read_csv(FICHIER_TRAIN, parse_dates=["datetime"])

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

        fichier_sortie = f"/home/fatma/elk-ussd-orange/ml-pipeline/data/shap_{serie}.csv"
        importance_moyenne.to_csv(fichier_sortie, header=["importance_shap"])
        print(f"Fichier ecrit : {fichier_sortie}")


if __name__ == "__main__":
    main()
