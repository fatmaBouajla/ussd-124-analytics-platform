import pandas as pd

from config_seuils import SEUIL_ALERTE_CA_PCT

FICHIER_RESULTAT = "/home/fatma/elk-ussd-orange/ml-pipeline/data/resultat_test_aveugle.csv"
FICHIER_PARQUET = "/home/fatma/elk-ussd-orange/ml-pipeline/data/cdr_clean.parquet"
FICHIER_SORTIE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/recommandations_alertes_test_aveugle.csv"

SEUIL_ECART_TAUX_ECHEC = 0.15


def calculer_ecart_pct(ligne):
    if ligne["ca_reel_prevu"] > 1:
        return round((ligne["ca_reel_ecart"] / ligne["ca_reel_prevu"]) * 100, 1)
    return 0.0


def main():
    resultat = pd.read_csv(FICHIER_RESULTAT, parse_dates=["datetime"])
    df_cdr = pd.read_parquet(FICHIER_PARQUET)

    resultat["ca_reel_ecart_pct"] = resultat.apply(calculer_ecart_pct, axis=1)

    en_alerte_ca = resultat["ca_reel_ecart_pct"].abs() >= SEUIL_ALERTE_CA_PCT
    en_alerte_taux_echec = resultat["taux_echec_ecart"] >= SEUIL_ECART_TAUX_ECHEC
    en_alerte = resultat[en_alerte_ca | en_alerte_taux_echec].copy()

    en_alerte["type_alerte"] = "ca"
    en_alerte.loc[en_alerte_taux_echec[en_alerte_ca | en_alerte_taux_echec].values & ~en_alerte_ca[en_alerte_ca | en_alerte_taux_echec].values, "type_alerte"] = "taux_echec"
    en_alerte.loc[en_alerte_ca[en_alerte_ca | en_alerte_taux_echec].values & en_alerte_taux_echec[en_alerte_ca | en_alerte_taux_echec].values, "type_alerte"] = "ca_et_taux_echec"

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

        resultats.append({
            "datetime": debut,
            "type_alerte": ligne["type_alerte"],
            "ca_ecart_pct": ligne["ca_reel_ecart_pct"],
            "ca_ecart_dt": ligne["ca_reel_ecart"],
            "taux_echec_reel": ligne["taux_echec_reel"],
            "taux_echec_prevu": ligne["taux_echec_prevu"],
            "taux_echec_ecart": ligne["taux_echec_ecart"],
            "nb_echecs_creneau": len(echecs_creneau),
            "cause_dominante": cause_dominante,
            "part_cause_dominante_pct": round(part * 100, 1),
        })

    resultats_df = pd.DataFrame(resultats).sort_values("datetime")
    resultats_df.to_csv(FICHIER_SORTIE, index=False)
    print(f"Alertes CA : {(resultats_df['type_alerte'] == 'ca').sum()}")
    print(f"Alertes taux echec : {(resultats_df['type_alerte'] == 'taux_echec').sum()}")
    print(f"Alertes CA et taux echec : {(resultats_df['type_alerte'] == 'ca_et_taux_echec').sum()}")
    print(f"Total : {len(resultats_df)}")
    print(f"Fichier ecrit : {FICHIER_SORTIE}")


if __name__ == "__main__":
    main()
