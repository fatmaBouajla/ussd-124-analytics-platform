import pandas as pd



FICHIER_ENTREE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/ml_dataset_30min.csv"

FICHIER_TRAIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_train.csv"

FICHIER_TEST = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_test.csv"

FICHIER_INCIDENT = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_incident.csv"



DEBUT_INCIDENT = "2026-07-08 12:00:00"

FIN_INCIDENT = "2026-07-09 11:30:00"



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





def main():

    df = pd.read_csv(FICHIER_ENTREE, parse_dates=["datetime"])

    df = ajouter_features(df)

    colonnes_obligatoires = [f"{s}_moins_24h" for s in SERIES] + [f"{s}_cible" for s in SERIES]
    df = df.dropna(subset=colonnes_obligatoires)

    est_incident = (df["datetime"] >= DEBUT_INCIDENT) & (df["datetime"] <= FIN_INCIDENT)
    df_incident = df[est_incident]
    df_normal = df[~est_incident]

    coupure = int(len(df_normal) * 0.8)
    df_train = df_normal.iloc[:coupure]
    df_test = df_normal.iloc[coupure:]

    df_train.to_csv(FICHIER_TRAIN, index=False)
    df_test.to_csv(FICHIER_TEST, index=False)
    df_incident.to_csv(FICHIER_INCIDENT, index=False)

    print(f"Lignes normales : {len(df_normal)}")
    print(f"Train : {len(df_train)} ({df_train['datetime'].min()} -> {df_train['datetime'].max()})")
    print(f"Test  : {len(df_test)} ({df_test['datetime'].min()} -> {df_test['datetime'].max()})")
    print(f"Incident : {len(df_incident)}")


if __name__ == "__main__":
    main()
