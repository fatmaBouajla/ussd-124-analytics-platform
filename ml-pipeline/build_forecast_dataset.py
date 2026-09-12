import pandas as pd

FICHIER_ENTREE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/ml_dataset_30min.csv"
FICHIER_TRAIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_train.csv"
FICHIER_TEST = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_test.csv"
FICHIER_INCIDENT = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_incident.csv"


COUPURE_CALIBRATION = "2026-07-25 23:59:59"

# Incidents historiques connus a exclure de la calibration et a isoler

LISTE_INCIDENTS_HISTORIQUES = [
    {"nom": "incident_08_07", "debut": "2026-07-08 12:00:00", "fin": "2026-07-09 11:30:00"},
]

SERIES = ["ca_reel", "nb_transactions", "taux_echec"]


def ajouter_features(df):
    for serie in SERIES:
        df[f"{serie}_moins_1h"] = df[serie].shift(2)
        df[f"{serie}_moins_2h"] = df[serie].shift(4)
        df[f"{serie}_moins_24h"] = df[serie].shift(48)
        df[f"{serie}_moyenne_3h"] = df[serie].shift(1).rolling(6).mean()
        df[f"{serie}_moyenne_6h"] = df[serie].shift(1).rolling(12).mean()
        df[f"{serie}_cible"] = df[serie].shift(-1)
    return df


def masque_incidents_historiques(df):
   
    masque = pd.Series(False, index=df.index)
    for incident in LISTE_INCIDENTS_HISTORIQUES:
        masque |= (df["datetime"] >= incident["debut"]) & (df["datetime"] <= incident["fin"])
    return masque


def main():
    df = pd.read_csv(FICHIER_ENTREE, parse_dates=["datetime"])
    df = ajouter_features(df)

    colonnes_obligatoires = [f"{s}_moins_24h" for s in SERIES] + [f"{s}_cible" for s in SERIES]
    df = df.dropna(subset=colonnes_obligatoires)

    est_incident_historique = masque_incidents_historiques(df)
    df_incident = df[est_incident_historique]
    df_restant = df[~est_incident_historique]

    est_calibration = df_restant["datetime"] <= COUPURE_CALIBRATION
    df_train = df_restant[est_calibration]
    df_test = df_restant[~est_calibration]

    df_train.to_csv(FICHIER_TRAIN, index=False)
    df_test.to_csv(FICHIER_TEST, index=False)
    df_incident.to_csv(FICHIER_INCIDENT, index=False)

    print(f"Coupure de calibration : {COUPURE_CALIBRATION}")
    print(f"Incidents historiques exclus : {[i['nom'] for i in LISTE_INCIDENTS_HISTORIQUES]}\n")

    print(f"Calibration (train) : {len(df_train)} lignes "
          f"({df_train['datetime'].min()} -> {df_train['datetime'].max()})")
    print(f"Test aveugle        : {len(df_test)} lignes "
          f"({df_test['datetime'].min()} -> {df_test['datetime'].max()})")
    print(f"Validation retrospective : {len(df_incident)} lignes "
          f"({df_incident['datetime'].min() if len(df_incident) else 'N/A'} -> "
          f"{df_incident['datetime'].max() if len(df_incident) else 'N/A'})")


if __name__ == "__main__":
    main()
