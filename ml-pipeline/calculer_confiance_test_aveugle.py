import pandas as pd

from feature_utils import SERIES, colonnes_finales

FICHIER_TRAIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_train.csv"
FICHIER_TEST_AVEUGLE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_test.csv"
FICHIER_SORTIE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/confiance_test_aveugle.csv"


def main():
    normal = pd.read_csv(FICHIER_TRAIN)
    test_aveugle = pd.read_csv(FICHIER_TEST_AVEUGLE, parse_dates=["datetime"])
    resultat = test_aveugle[["datetime"]].copy()

    for serie in SERIES:
        colonnes = colonnes_finales(serie)
        bornes = {c: (normal[c].min(), normal[c].max()) for c in colonnes}

        liste_hors_zone = []
        for _, ligne in test_aveugle.iterrows():
            hors_zone = [c for c in colonnes if ligne[c] < bornes[c][0] or ligne[c] > bornes[c][1]]
            liste_hors_zone.append(", ".join(hors_zone) if hors_zone else "")

        resultat[f"{serie}_hors_zone"] = [1 if v else 0 for v in liste_hors_zone]
        resultat[f"{serie}_features_hors_zone"] = liste_hors_zone

    resultat.to_csv(FICHIER_SORTIE, index=False)
    print(f"Fichier ecrit : {FICHIER_SORTIE}")


if __name__ == "__main__":
    main()
