import pandas as pd



INPUT_CSV_PATH = "/home/fatma/elk-ussd-orange/ml-pipeline/data/offres.csv"

OUTPUT_CSV_PATH = "/home/fatma/elk-ussd-orange/ml-pipeline/data/offres_classement.csv"



VOLUME_MIN = 100  # seuil en dessous duquel le classement est peu fiable





def main():

    df = pd.read_csv(INPUT_CSV_PATH)



    agg = df.groupby(["offer_code", "offer_name"]).agg(

        ca_total=("ca_offre", "sum"),

        nb_souscriptions_total=("nb_souscriptions", "sum"),

        nb_succes_total=("nb_succes", "sum"),

        nb_echecs_total=("nb_echecs", "sum"),

    ).reset_index()



    agg["taux_succes_global"] = (

        agg["nb_succes_total"] / agg["nb_souscriptions_total"]

    ).round(4)



    agg["fiable"] = agg["nb_souscriptions_total"] >= VOLUME_MIN



    print("=== Top 5 par CA ===")

    print(agg.sort_values("ca_total", ascending=False).head(5)[

        ["offer_code", "offer_name", "ca_total", "fiable"]])



    print("\n=== Top 5 par volume (offre la plus demandee) ===")

    print(agg.sort_values("nb_souscriptions_total", ascending=False).head(5)[
        ["offer_code", "offer_name", "nb_souscriptions_total", "fiable"]])

    print("\n=== Top 5 meilleur taux de succes (fiables uniquement) ===")
    print(agg[agg["fiable"]].sort_values("taux_succes_global", ascending=False).head(5)[
        ["offer_code", "offer_name", "taux_succes_global", "nb_souscriptions_total"]])

    print("\n=== Top 5 plus d'echecs ===")
    print(agg.sort_values("nb_echecs_total", ascending=False).head(5)[
        ["offer_code", "offer_name", "nb_echecs_total", "fiable"]])

    agg.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nClassement complet ecrit : {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
