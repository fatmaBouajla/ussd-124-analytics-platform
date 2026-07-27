import pandas as pd



FICHIER_CLASSEMENT = "/home/fatma/elk-ussd-orange/ml-pipeline/data/offres_classement.csv"

FICHIER_TENDANCE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/tendance_offres.csv"

FICHIER_SORTIE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/indice_performance_offres.csv"



POIDS_CA = 0.35

POIDS_SUCCES = 0.30

POIDS_VOLUME = 0.20

POIDS_TENDANCE = 0.15





def main():

    classement = pd.read_csv(FICHIER_CLASSEMENT)

    tendance = pd.read_csv(FICHIER_TENDANCE)



    df = classement.merge(

        tendance[["offer_code", "variation_ca_pct", "fiable"]],

        on="offer_code", suffixes=("", "_tendance")

    )

    df = df.rename(columns={"fiable_tendance": "fiable_tendance"})



    df_fiable = df[df["fiable"]].copy()



    df_fiable["rang_ca"] = df_fiable["ca_total"].rank(pct=True)

    df_fiable["rang_succes"] = df_fiable["taux_succes_global"].rank(pct=True)

    df_fiable["rang_volume"] = df_fiable["nb_souscriptions_total"].rank(pct=True)

    df_fiable["rang_tendance"] = df_fiable["variation_ca_pct"].rank(pct=True)



    df_fiable["indice_performance"] = (

        df_fiable["rang_ca"] * POIDS_CA

        + df_fiable["rang_succes"] * POIDS_SUCCES

        + df_fiable["rang_volume"] * POIDS_VOLUME

        + df_fiable["rang_tendance"] * POIDS_TENDANCE

    ) * 100



    df_fiable = df_fiable.sort_values("indice_performance", ascending=False)

    df_fiable.to_csv(FICHIER_SORTIE, index=False)



    print("=== Classement complet - Indice de performance de l'offre ===")

    print(df_fiable[["offer_code", "offer_name", "indice_performance",

                      "ca_total", "taux_succes_global", "nb_souscriptions_total",

                      "variation_ca_pct"]].round(1).to_string(index=False))



    print(f"\nFichier ecrit : {FICHIER_SORTIE}")

    print(f"\nNombre d'offres non incluses (non fiables pour tendance) : {len(df) - len(df_fiable)}")





if __name__ == "__main__":

    main()
