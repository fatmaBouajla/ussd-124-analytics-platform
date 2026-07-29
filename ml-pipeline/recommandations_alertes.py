import pandas as pd



FICHIER_INCIDENT_RESULTAT = "/home/fatma/elk-ussd-orange/ml-pipeline/data/resultat_incident.csv"

FICHIER_PARQUET = "/home/fatma/elk-ussd-orange/ml-pipeline/data/cdr_clean.parquet"

FICHIER_SORTIE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/recommandations_alertes.csv"



SEUIL_ALERTE_CA = 1000



ACTIONS_PAR_CAUSE = {

    "Exception on update main balance (Commande IN updatebalanceAnddate KO)":

        "Escalade equipe technique IN/BSCS - verifier capacite/timeout du service "

        "de mise a jour de solde, notamment pour les offres a gros volume de donnees.",

    "Other BuyOption ErrCode :1":

        "Investigation infrastructure immediate - verifier logs et disponibilite "

        "du sous-systeme BuyOption au moment precis de l'alerte.",

    "Utilisateur non autorise pour acheter l'option":

        "Verifier les regles de provisioning/eligibilite - probablement pas un "

        "incident technique.",

    "Option/bonus non provisionnee sur la base OTNWS":

        "Verifier la  de l'offre dans la base OTNWS.",
    "Nbre max d'utilisation de l'option est atteint":
        "Limite metier normale - verifier si le seuil est toujours pertinent.",
    "API OpenCode KO fonctionnel":
        "Verifier la sante/disponibilite de l'API OpenCode.",
    "Service non implemente":
        "Verifier l'activation du service pour ce code.",
}


def action_pour_cause(cause):
    if cause in ACTIONS_PAR_CAUSE:
        return ACTIONS_PAR_CAUSE[cause]
    return "Cause non repertoriee - investigation manuelle necessaire."


def main():
    incident = pd.read_csv(FICHIER_INCIDENT_RESULTAT, parse_dates=["datetime"])
    df_cdr = pd.read_parquet(FICHIER_PARQUET)

    en_alerte = incident[incident["ca_reel_ecart"].abs() >= SEUIL_ALERTE_CA].copy()

    resultats = []
    for _, ligne in en_alerte.iterrows():
        debut = ligne["datetime"]
        fin = debut + pd.Timedelta(minutes=30)

        echecs_creneau = df_cdr[
            (df_cdr["timestamp"] >= debut) & (df_cdr["timestamp"] < fin) &
            (df_cdr["event_type"] == "transaction_failed")
        ]

        if len(echecs_creneau) == 0:
            cause_dominante = "aucun echec identifie sur ce creneau"
            part = 0.0
        else:
            compte = echecs_creneau["error_description"].value_counts()
            cause_dominante = compte.index[0]
            part = compte.iloc[0] / len(echecs_creneau)

        action = action_pour_cause(cause_dominante)

        resultats.append({
            "datetime": debut,
            "ca_ecart": ligne["ca_reel_ecart"],
            "nb_echecs_creneau": len(echecs_creneau),
            "cause_dominante": cause_dominante,
            "part_cause_dominante_pct": round(part * 100, 1),
            "action_recommandee": action,
        })

    resultats_df = pd.DataFrame(resultats)
    resultats_df.to_csv(FICHIER_SORTIE, index=False)

    print("=== Recommandations par creneau en alerte ===")
    for _, l in resultats_df.iterrows():
        print(f"\n{l['datetime']} (ecart CA: {l['ca_ecart']:.0f} DT)")
        print(f"  Cause dominante: {l['cause_dominante']} ({l['part_cause_dominante_pct']:.0f}% des echecs)")
        print(f"  Action recommandee: {l['action_recommandee']}")

    print(f"\nFichier ecrit : {FICHIER_SORTIE}")


if __name__ == "__main__":
    main()
