import pandas as pd

import numpy as np



FICHIER_ENTREE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/offres.csv"

FICHIER_SORTIE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/tendance_offres.csv"



JOURS_INCIDENT = ["2026-07-08", "2026-07-09"]

SEUIL_TENDANCE = 0.10

JOURS_MIN_FIABLE = 10

VOLUME_MIN_TOTAL = 1000

NB_JOURS_COMPARAISON = 5





def comparer_debut_fin(valeurs):

    if len(valeurs) < NB_JOURS_COMPARAISON * 2:

        debut = valeurs[:len(valeurs) // 2]

        fin = valeurs[len(valeurs) // 2:]

    else:

        debut = valeurs[:NB_JOURS_COMPARAISON]

        fin = valeurs[-NB_JOURS_COMPARAISON:]



    moyenne_debut = debut.mean()

    moyenne_fin = fin.mean()



    if moyenne_debut == 0:

        return 0.0, moyenne_debut, moyenne_fin



    variation = (moyenne_fin - moyenne_debut) / moyenne_debut

    return variation, moyenne_debut, moyenne_fin





def classer_tendance(variation_relative):

    if variation_relative > SEUIL_TENDANCE:

        return "hausse"

    elif variation_relative < -SEUIL_TENDANCE:

        return "baisse"

    else:

        return "stable"





def main():

    df = pd.read_csv(FICHIER_ENTREE, parse_dates=["date"])

    df = df[~df["date"].dt.strftime("%Y-%m-%d").isin(JOURS_INCIDENT)]



    resultats = []

    for offer_code, groupe in df.groupby("offer_code"):

        groupe = groupe.sort_values("date")

        offer_name = groupe["offer_name"].iloc[0]

        volume_total = groupe["nb_souscriptions"].sum()



        variation_volume, _, _ = comparer_debut_fin(groupe["nb_souscriptions"].values)

        variation_ca, ca_debut, ca_fin = comparer_debut_fin(groupe["ca_offre"].values)



        fiable = len(groupe) >= JOURS_MIN_FIABLE and volume_total >= VOLUME_MIN_TOTAL



        resultats.append({

            "offer_code": offer_code,

            "offer_name": offer_name,

            "nb_jours_presents": len(groupe),

            "volume_total": int(volume_total),

            "fiable": fiable,

            "variation_volume_pct": round(variation_volume * 100, 1),

            "tendance_volume": classer_tendance(variation_volume) if fiable else "non_evaluable",

            "variation_ca_pct": round(variation_ca * 100, 1),

            "tendance_ca": classer_tendance(variation_ca) if fiable else "non_evaluable",

            "ca_moyen_debut": round(ca_debut, 1),

            "ca_moyen_fin": round(ca_fin, 1),

        })



    resultats_df = pd.DataFrame(resultats).sort_values("variation_ca_pct", ascending=False)

    resultats_df.to_csv(FICHIER_SORTIE, index=False)



    print("=== Offres fiables en forte hausse (CA) ===")

    print(resultats_df[(resultats_df["tendance_ca"] == "hausse") & (resultats_df["fiable"])][

        ["offer_code", "offer_name", "volume_total", "variation_ca_pct"]].to_string(index=False))



    print("\n=== Offres fiables en forte baisse (CA) ===")

    print(resultats_df[(resultats_df["tendance_ca"] == "baisse") & (resultats_df["fiable"])][

        ["offer_code", "offer_name", "volume_total", "variation_ca_pct"]].to_string(index=False))



    print(f"\nNombre d'offres non evaluables (volume/presence insuffisants) : {(~resultats_df['fiable']).sum()}")

    print(f"Fichier ecrit : {FICHIER_SORTIE}")





if __name__ == "__main__":

    main()
