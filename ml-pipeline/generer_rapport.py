import pandas as pd



from config_seuils import SEUIL_ALERTE_CA_PCT



FICHIER_INCIDENT = "/home/fatma/elk-ussd-orange/ml-pipeline/data/resultat_incident.csv"

FICHIER_CONFIANCE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/confiance_incident.csv"

FICHIER_SHAP_CA = "/home/fatma/elk-ussd-orange/ml-pipeline/data/shap_ca_reel.csv"

FICHIER_CLASSEMENT = "/home/fatma/elk-ussd-orange/ml-pipeline/data/offres_classement.csv"

FICHIER_RAPPORT = "/home/fatma/elk-ussd-orange/ml-pipeline/data/rapport_incident.txt"





def calculer_ecart_pct(ligne):

    if ligne["ca_reel_prevu"] > 1:

        return round((ligne["ca_reel_ecart"] / ligne["ca_reel_prevu"]) * 100, 1)

    return 0.0





def main():

    if SEUIL_ALERTE_CA_PCT is None:

        raise ValueError(

            "SEUIL_ALERTE_CA_PCT n'est pas encore defini dans config_seuils.py. "

            "Lancer analyser_distribution_ecarts.py, choisir un seuil avec Fatma, "

            "puis renseigner la valeur avant de generer le rapport."

        )



    incident = pd.read_csv(FICHIER_INCIDENT, parse_dates=["datetime"])
    confiance = pd.read_csv(FICHIER_CONFIANCE, parse_dates=["datetime"])
    shap_ca = pd.read_csv(FICHIER_SHAP_CA, index_col=0)
    classement = pd.read_csv(FICHIER_CLASSEMENT)

    donnees = incident.merge(confiance, on="datetime")
    donnees["ca_reel_ecart_pct"] = donnees.apply(calculer_ecart_pct, axis=1)

    lignes = []
    lignes.append("=== Rapport automatique - Periode analysee ===\n")

    en_alerte = donnees[donnees["ca_reel_ecart_pct"].abs() >= SEUIL_ALERTE_CA_PCT]
    duree_heures = len(en_alerte) * 0.5
    ecart_cumule = en_alerte["ca_reel_ecart"].sum()

    lignes.append(
        f"Seuil d'alerte applique : ecart de {SEUIL_ALERTE_CA_PCT:.0f}% ou plus "
        f"entre CA reel et CA prevu."
    )
    lignes.append(
        f"Nombre de creneaux en alerte : {len(en_alerte)} sur {len(donnees)} "
        f"({duree_heures:.1f}h cumulees)."
    )
    lignes.append(
        f"Ecart cumule de chiffre d'affaires sur ces creneaux : {ecart_cumule:.0f} DT "
        f"(manque a gagner si negatif)."
    )

    nb_hors_zone = donnees["ca_reel_hors_zone"].sum()
    if nb_hors_zone > 0:
        lignes.append(
            f"\nAttention : {nb_hors_zone} predictions de CA ont ete generees hors de la "
            f"zone de confiance du modele (situation inedite en apprentissage) - "
            f"a interpreter avec prudence sur ces creneaux."
        )

    lignes.append("\nDetail des creneaux en alerte :")
    for _, ligne in en_alerte.iterrows():
        ecart = ligne["ca_reel_ecart"]
        ecart_pct = ligne["ca_reel_ecart_pct"]
        sens = "en dessous" if ecart < 0 else "au dessus"
        note = " [hors zone de confiance]" if ligne["ca_reel_hors_zone"] == 1 else ""
        lignes.append(
            f"- {ligne['datetime']} : CA reel {ligne['ca_reel_reel']:.0f} DT, "
            f"{sens} du CA attendu ({ligne['ca_reel_prevu']:.0f} DT), "
            f"ecart {ecart:.0f} DT ({ecart_pct:+.1f}%).{note}"
        )

    facteur_principal = shap_ca.index[0]
    lignes.append(f"\nFacteur explicatif principal des variations de CA : {facteur_principal}")

    top_ca = classement.sort_values("ca_total", ascending=False).iloc[0]
    top_volume = classement.sort_values("nb_souscriptions_total", ascending=False).iloc[0]
    lignes.append(
        f"\nOffre generant le plus de CA sur la periode : {top_ca['offer_code']} "
        f"({top_ca['offer_name']}), {top_ca['ca_total']:.0f} DT."
    )
    lignes.append(
        f"Offre la plus demandee : {top_volume['offer_code']} "
        f"({top_volume['offer_name']}), {top_volume['nb_souscriptions_total']:.0f} souscriptions."
    )

    texte = "\n".join(lignes)
    print(texte)

    with open(FICHIER_RAPPORT, "w", encoding="utf-8") as f:
        f.write(texte)
    print(f"\nRapport ecrit : {FICHIER_RAPPORT}")


if __name__ == "__main__":
    main()
