import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from feature_utils import SERIES, colonnes_finales

FICHIER_TRAIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_train.csv"
FICHIER_TEST_AVEUGLE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_test.csv"
FICHIER_RESULTAT = "/home/fatma/elk-ussd-orange/ml-pipeline/data/resultat_test_aveugle.csv"


def main():
    train = pd.read_csv(FICHIER_TRAIN, parse_dates=["datetime"])
    test_aveugle = pd.read_csv(FICHIER_TEST_AVEUGLE, parse_dates=["datetime"])
    resultat = test_aveugle[["datetime"]].copy()

    for serie in SERIES:
        cible = f"{serie}_cible"
        colonnes = colonnes_finales(serie)

        modele = RandomForestRegressor(n_estimators=200, random_state=42)
        modele.fit(train[colonnes], train[cible])
        prediction = modele.predict(test_aveugle[colonnes])

        resultat[f"{serie}_reel"] = test_aveugle[cible].values
        resultat[f"{serie}_prevu"] = prediction.round(2)
        resultat[f"{serie}_ecart"] = (resultat[f"{serie}_reel"] - resultat[f"{serie}_prevu"]).round(2)

    resultat.to_csv(FICHIER_RESULTAT, index=False)
    print(f"Entrainement : {len(train)} lignes ({train['datetime'].min()} -> {train['datetime'].max()})")
    print(f"Test aveugle : {len(test_aveugle)} lignes ({test_aveugle['datetime'].min()} -> {test_aveugle['datetime'].max()})")
    print(f"Fichier ecrit : {FICHIER_RESULTAT}")


if __name__ == "__main__":
    main()
