import pandas as pd



INPUT_PARQUET_PATH = "/home/fatma/elk-ussd-orange/ml-pipeline/data/cdr_clean.parquet"

OUTPUT_CSV_PATH = "/home/fatma/elk-ussd-orange/ml-pipeline/data/kpi_global.csv"



def main():

    df = pd.read_parquet(INPUT_PARQUET_PATH)

    df["date"] = df["timestamp"].dt.date



    rows = []

    for date, group in df.groupby("date"):

        nb_evenements_total = len(group)

        nb_consultations = (group["event_type"] == "consultation").sum()

        nb_succes = (group["event_type"] == "transaction_success").sum()

        nb_echecs = (group["event_type"] == "transaction_failed").sum()

        nb_unknown = (group["event_type"] == "unknown").sum()

        nb_transactions = nb_succes + nb_echecs



        taux_succes = nb_succes / nb_transactions if nb_transactions > 0 else None

        taux_echec = nb_echecs / nb_transactions if nb_transactions > 0 else None



        ca_reel = group.loc[group["event_type"] == "transaction_success", "price"].sum()



        offres_utilisees = group.loc[

            group["event_type"].isin(["transaction_success", "transaction_failed"]),

            "offer"

        ]

        nb_offres_distinctes = offres_utilisees.nunique()



        rows.append({

            "date": date,

            "nb_evenements_total": nb_evenements_total,

            "nb_consultations": nb_consultations,

            "nb_transactions": nb_transactions,

            "nb_succes": nb_succes,

            "nb_echecs": nb_echecs,

            "nb_unknown": nb_unknown,

            "taux_succes": round(taux_succes, 4) if taux_succes is not None else None,

            "taux_echec": round(taux_echec, 4) if taux_echec is not None else None,

            "ca_reel": round(ca_reel, 2),

            "nb_offres_distinctes": nb_offres_distinctes,

        })



    kpi_df = pd.DataFrame(rows).sort_values("date")

    kpi_df.to_csv(OUTPUT_CSV_PATH, index=False)



    print(f"Dataset KPI global ecrit : {OUTPUT_CSV_PATH}")

    print(f"Nombre de jours : {len(kpi_df)}")

    print("\nApercu :")

    print(kpi_df.to_string(index=False))



    print("\n--- Verification de coherence ")

    print(f"Somme nb_succes    : {kpi_df['nb_succes'].sum()}  ")

    print(f"Somme nb_echecs    : {kpi_df['nb_echecs'].sum()} ")

    print(f"Somme nb_consultations : {kpi_df['nb_consultations'].sum()}  ")

    print(f"Somme nb_unknown   : {kpi_df['nb_unknown'].sum()}  ")



if __name__ == "__main__":

    main()
