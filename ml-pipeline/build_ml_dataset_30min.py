import pandas as pd



FICHIER_ENTREE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/cdr_clean.parquet"

FICHIER_SORTIE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/ml_dataset_30min.csv"





def calculer_ligne(creneau, donnees):

    succes = (donnees["event_type"] == "transaction_success").sum()

    echecs = (donnees["event_type"] == "transaction_failed").sum()

    total_transactions = succes + echecs

    taux_echec = echecs / total_transactions if total_transactions > 0 else 0.0

    ca = donnees.loc[donnees["event_type"] == "transaction_success", "price"].sum()



    return {

        "datetime": creneau,

        "heure_du_jour": creneau.hour,

        "jour_semaine": creneau.dayofweek,

        "est_weekend": int(creneau.dayofweek >= 5),

        "nb_evenements": len(donnees),

        "nb_consultations": (donnees["event_type"] == "consultation").sum(),

        "nb_transactions": total_transactions,

        "nb_succes": succes,

        "nb_echecs": echecs,

        "taux_echec": round(taux_echec, 4),
        "ca_reel": round(ca, 2),
    }


def main():
    df = pd.read_parquet(FICHIER_ENTREE)
    df["creneau"] = df["timestamp"].dt.floor("30min")

    lignes = [calculer_ligne(creneau, groupe) for creneau, groupe in df.groupby("creneau")]

    resultat = pd.DataFrame(lignes).sort_values("datetime").reset_index(drop=True)

   
    resultat["position_periode"] = range(len(resultat))

    resultat.to_csv(FICHIER_SORTIE, index=False)

    print(f"Fichier ecrit : {FICHIER_SORTIE}")
    print(f"Nombre de lignes : {len(resultat)}")
    print(f"Total succes : {resultat['nb_succes'].sum()} (attendu 1189963)")
    print(f"Total echecs : {resultat['nb_echecs'].sum()} (attendu 570141)")
    print(f"Total CA : {round(resultat['ca_reel'].sum(), 2)}")


if __name__ == "__main__":
    main()
