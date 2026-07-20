import pandas as pd

INPUT_PARQUET_PATH = "/home/fatma/elk-ussd-orange/ml-pipeline/data/cdr_clean.parquet"
OUTPUT_CSV_PATH = "/home/fatma/elk-ussd-orange/ml-pipeline/data/offres.csv"


def main():
    df = pd.read_parquet(INPUT_PARQUET_PATH)
    df["date"] = df["timestamp"].dt.date

    df_tx = df[df["event_type"].isin(["transaction_success", "transaction_failed"])].copy()

    rows = []
    for (date, offer), group in df_tx.groupby(["date", "offer"]):
        offer_name = group["offer_name"].iloc[0]
        nb_souscriptions = len(group)
        nb_succes = (group["event_type"] == "transaction_success").sum()
        nb_echecs = (group["event_type"] == "transaction_failed").sum()
        taux_succes = nb_succes / nb_souscriptions if nb_souscriptions > 0 else None
        ca_offre = group.loc[group["event_type"] == "transaction_success", "price"].sum()

        rows.append({
            "date": date,
            "offer_code": offer,
            "offer_name": offer_name,
            "nb_souscriptions": nb_souscriptions,
            "nb_succes": nb_succes,
            "nb_echecs": nb_echecs,
            "taux_succes": round(taux_succes, 4) if taux_succes is not None else None,
            "ca_offre": round(ca_offre, 2),
        })

    offres_df = pd.DataFrame(rows).sort_values(["date", "nb_souscriptions"], ascending=[True, False])
    offres_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"Dataset offres ecrit : {OUTPUT_CSV_PATH}")
    print(f"Nombre de lignes (jour x offre) : {len(offres_df)}")
    print(f"Nombre d'offres distinctes : {offres_df['offer_code'].nunique()}")

    print("\n--- Verification de coherence ---")
    print(f"Somme nb_succes  : {offres_df['nb_succes'].sum()}  (attendu 1189963)")
    print(f"Somme nb_echecs  : {offres_df['nb_echecs'].sum()}  (attendu 570141)")

    print("\nTop 10 offres par nb total de souscriptions (toute periode confondue) :")
    top = offres_df.groupby(["offer_code", "offer_name"])["nb_souscriptions"].sum().sort_values(ascending=False).head(10)
    print(top)


if __name__ == "__main__":
    main()
