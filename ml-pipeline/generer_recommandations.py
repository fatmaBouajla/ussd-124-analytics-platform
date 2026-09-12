import pandas as pd

from sklearn.ensemble import IsolationForest

from config_seuils import (
    SEUIL_TAUX_CRITIQUE, SEUIL_TAUX_VIGILANCE, SEUIL_HAUSSE, SEUIL_BAISSE,
    SEUIL_CAUSE_DOMINANTE, CONTAMINATION_ISOLATION_FOREST,
)
from classification_causes_utils import statut_cause, charger_classification

FICHIER_INDICE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/indice_performance_offres.csv"
FICHIER_PARQUET = "/home/fatma/elk-ussd-orange/ml-pipeline/data/cdr_clean.parquet"
FICHIER_SORTIE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/recommandations_offres.csv"


def obtenir_cause_dominante(df_cdr, offer_code):
    echecs = df_cdr[(df_cdr["offer"] == offer_code) & (df_cdr["event_type"] == "transaction_failed")]
    if len(echecs) == 0:
        return None, 0.0
    compte = echecs["error_description"].value_counts()
    cause_principale = compte.index[0]
    part = compte.iloc[0] / len(echecs)
    return cause_principale, part


def detecter_anomalies_statistiques(df):
    features = ["ca_total", "taux_succes_global", "nb_souscriptions_total", "variation_ca_pct"]
    x = df[features].copy()
    x = (x - x.mean()) / x.std()

    modele = IsolationForest(contamination=CONTAMINATION_ISOLATION_FOREST, random_state=42)
    predictions = modele.fit_predict(x)
    scores = modele.decision_function(x)

    df["anomalie_statistique"] = (predictions == -1).astype(int)
    df["score_anomalie"] = scores.round(3)
    return df


def generer_constat(ligne, cause, part_cause, classification):
    
    taux = ligne["taux_succes_global"]
    variation = ligne["variation_ca_pct"]
    ca = ligne["ca_total"]

    if taux < SEUIL_TAUX_CRITIQUE:
        statut = statut_cause(cause, classification) if cause else "a_verifier"

        if statut == "technique_ponctuelle" and part_cause >= SEUIL_CAUSE_DOMINANTE:
            texte = (f"Taux de succes critique ({taux*100:.0f}%), fortement affecte "
                     f"par un incident technique confirme de la plateforme "
                     f"('{cause}', {part_cause*100:.0f}% des echecs) - pas un probleme "
                     f"isole a cette offre.")
            priorite = "technique_critique"

        elif statut == "metier_stable" and part_cause >= SEUIL_CAUSE_DOMINANTE:
            texte = (f"Taux de succes critique ({taux*100:.0f}%), mais majoritairement "
                     f"explique par un comportement client normal "
                     f"('{cause}', {part_cause*100:.0f}% des echecs) - pas un "
                     f"dysfonctionnement de la plateforme.")
            priorite = "taux_critique_non_technique"

        elif statut == "a_verifier" and cause is not None and part_cause >= SEUIL_CAUSE_DOMINANTE:
            texte = (f"Taux de succes critique ({taux*100:.0f}%), cause dominante non "
                     f"encore repertoriee ('{cause}', {part_cause*100:.0f}% des echecs) "
                     f"- a verifier manuellement.")
            priorite = "taux_critique_non_technique"

        else:
            texte = (f"Taux de succes critique ({taux*100:.0f}%) sans cause dominante "
                     f"claire (causes d'echec dispersees).")
            priorite = "technique_critique"

    elif taux < SEUIL_TAUX_VIGILANCE:
        texte = (f"Fiabilite en zone de vigilance ({taux*100:.0f}%), en dessous du "
                 f"niveau sain (>70%) sans etre critique.")
        priorite = "vigilance"

    elif variation > SEUIL_HAUSSE:
        texte = (f"Offre fiable (succes {taux*100:.0f}%) en forte croissance "
                 f"({variation:+.1f}% de CA).")
        priorite = "opportunite"

    elif variation < SEUIL_BAISSE:
        texte = (f"Baisse marquee ({variation:+.1f}%), CA total sur la periode : "
                 f"{ca:.0f} DT.")
        priorite = "marketing"

    else:
        texte = f"Performance stable (succes {taux*100:.0f}%, CA {ca:.0f} DT)."
        priorite = "neutre"

    return texte, priorite


def main():
    df = pd.read_csv(FICHIER_INDICE)
    df_cdr = pd.read_parquet(FICHIER_PARQUET)
    classification = charger_classification()

    df = detecter_anomalies_statistiques(df)

    resultats = []
    for _, ligne in df.iterrows():
        cause, part = obtenir_cause_dominante(df_cdr, ligne["offer_code"])
        texte, priorite = generer_constat(ligne, cause, part, classification)

        if ligne["anomalie_statistique"] == 1 and priorite in ("vigilance", "neutre"):
            texte += (" [Signal complementaire] Profil statistiquement atypique "
                      "par rapport a l'ensemble des offres (detection multivariee), "
                      "meme si aucun seuil individuel n'est franchi.")
            priorite = "anomalie_detectee"

        resultats.append({
            "offer_code": ligne["offer_code"],
            "offer_name": ligne["offer_name"],
            "indice_performance": ligne["indice_performance"],
            "anomalie_statistique": ligne["anomalie_statistique"],
            "cause_dominante": cause,
            "part_cause_dominante_pct": round(part * 100, 1),
            "priorite": priorite,
            "constat": texte,
        })

    resultats_df = pd.DataFrame(resultats)
    ordre = {"technique_critique": 0, "anomalie_detectee": 1,
             "taux_critique_non_technique": 2, "vigilance": 3,
             "opportunite": 4, "marketing": 5, "neutre": 6}
    resultats_df = resultats_df.sort_values(by="priorite", key=lambda x: x.map(ordre))
    resultats_df.to_csv(FICHIER_SORTIE, index=False)

    for priorite in ordre:
        sous = resultats_df[resultats_df["priorite"] == priorite]
        if len(sous) > 0:
            print(f"\n=== {priorite.upper()} ({len(sous)} offre(s)) ===")
            for _, l in sous.iterrows():
                print(f"\n{l['offer_code']} ({l['offer_name']}):")
                print(f"  {l['constat']}")

    print(f"\nFichier ecrit : {FICHIER_SORTIE}")


if __name__ == "__main__":
    main()
