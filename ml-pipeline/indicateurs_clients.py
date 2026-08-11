import pandas as pd



from config_seuils import SEUIL_TENTATIVES_REESSAI, SEUIL_TENTATIVES_BLOQUE



FICHIER_ENTREE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/cdr_clean.parquet"

FICHIER_SORTIE_30MIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/indicateurs_clients_30min.csv"

FICHIER_SORTIE_BLOQUES = "/home/fatma/elk-ussd-orange/ml-pipeline/data/clients_bloques.csv"





def calculer_indicateurs_30min(df):

    """

    Par creneau de 30 min (meme grain que ml_dataset_30min.csv, pour

    pouvoir croiser directement avec les creneaux en alerte CA) :

      - nb_clients_impactes : clients distincts avec au moins 1 echec

      - nb_clients_reessai : parmi eux, ceux avec >= SEUIL_TENTATIVES_REESSAI

        echecs DANS CE MEME CRENEAU (signal d'acharnement immediat)

    """

    df_echecs = df[df["event_type"] == "transaction_failed"].copy()

    df_echecs["creneau"] = df_echecs["timestamp"].dt.floor("30min")



    lignes = []

    for creneau, groupe in df_echecs.groupby("creneau"):

        nb_clients_impactes = groupe["subscriber"].nunique()

        tentatives_par_client = groupe.groupby("subscriber").size()
        nb_clients_reessai = (tentatives_par_client >= SEUIL_TENTATIVES_REESSAI).sum()

        lignes.append({
            "datetime": creneau,
            "nb_clients_impactes": nb_clients_impactes,
            "nb_clients_reessai": int(nb_clients_reessai),
        })

    resultat = pd.DataFrame(lignes).sort_values("datetime").reset_index(drop=True)
    return resultat


def calculer_clients_bloques(df):
    """
    Clients avec 0 succes ET au moins SEUIL_TENTATIVES_BLOQUE echecs sur
    TOUTE la periode disponible - signal fort d'un probleme persistant
    justifiant un contact support proactif.
    """
    df_tx = df[df["event_type"].isin(["transaction_success", "transaction_failed"])]

    agg = df_tx.groupby("subscriber").agg(
        nb_succes=("event_type", lambda x: (x == "transaction_success").sum()),
        nb_echecs=("event_type", lambda x: (x == "transaction_failed").sum()),
        premiere_tentative=("timestamp", "min"),
        derniere_tentative=("timestamp", "max"),
    ).reset_index()

    bloques = agg[
        (agg["nb_succes"] == 0) & (agg["nb_echecs"] >= SEUIL_TENTATIVES_BLOQUE)
    ].sort_values("nb_echecs", ascending=False)

    return bloques


def main():
    df = pd.read_parquet(FICHIER_ENTREE)

    indicateurs_30min = calculer_indicateurs_30min(df)
    indicateurs_30min.to_csv(FICHIER_SORTIE_30MIN, index=False)
    print(f"Fichier ecrit : {FICHIER_SORTIE_30MIN}")
    print(f"Nombre de creneaux avec au moins 1 echec : {len(indicateurs_30min)}")
    print(f"Pic de clients impactes sur un seul creneau : "
          f"{indicateurs_30min['nb_clients_impactes'].max()}")

    clients_bloques = calculer_clients_bloques(df)
    clients_bloques.to_csv(FICHIER_SORTIE_BLOQUES, index=False)
    print(f"\nFichier ecrit : {FICHIER_SORTIE_BLOQUES}")
    print(f"Nombre de clients bloques (0 succes, >={SEUIL_TENTATIVES_BLOQUE} echecs) : "
          f"{len(clients_bloques)}")
    if len(clients_bloques) > 0:
        print("\nTop 5 clients bloques (plus grand nombre d'echecs) :")
        print(clients_bloques.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
