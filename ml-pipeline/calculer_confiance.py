import pandas as pd

from feature_utils import SERIES, colonnes_finales

FICHIER_TRAIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_train.csv"
FICHIER_TEST = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_test.csv"
FICHIER_INCIDENT = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_incident.csv"
FICHIER_SORTIE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/confiance_incident.csv"


def main():
    train = pd.read_csv(FICHIER_TRAIN)
    test = pd.read_csv(FICHIER_TEST)
    normal = pd.concat([train, test], ignore_index=True)

    incident = pd.read_csv(FICHIER_INCIDENT, parse_dates=["datetime"])

    resultat = incident[["datetime"]].copy()

    for serie in SERIES:
        colonnes = colonnes_finales(serie)
        bornes = {c: (normal[c].min(), normal[c].max()) for c in colonnes}

        liste_hors_zone = []
        for _, ligne in incident.iterrows():
            hors_zone = [c for c in colonnes if ligne[c] < bornes[c][0] or ligne[c] > bornes[c][1]]
            liste_hors_zone.append(", ".join(hors_zone) if hors_zone else "")

        resultat[f"{serie}_hors_zone"] = [1 if v else 0 for v in liste_hors_zone]
        resultat[f"{serie}_features_hors_zone"] = liste_hors_zone

    resultat.to_csv(FICHIER_SORTIE, index=False)

    print(f"Fichier ecrit : {FICHIER_SORTIE}")
    for serie in SERIES:
        nb = resultat[f"{serie}_hors_zone"].sum()
        print(f"{serie} : {nb} creneaux hors zone de confiance sur {len(resultat)}")


if __name__ == "__main__":
    main()
