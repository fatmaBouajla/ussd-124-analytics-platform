import pandas as pd

from sklearn.ensemble import IsolationForest



FICHIER_INDICE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/indice_performance_offres.csv"

FICHIER_PARQUET = "/home/fatma/elk-ussd-orange/ml-pipeline/data/cdr_clean.parquet"

FICHIER_SORTIE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/recommandations_offres.csv"



SEUIL_TAUX_CRITIQUE = 0.50

SEUIL_TAUX_VIGILANCE = 0.70

SEUIL_HAUSSE = 20.0

SEUIL_BAISSE = -30.0

CAUSE_CHRONIQUE = "Exception on update main balance (Commande IN updatableAndate KO)"

SEUIL_CAUSE_DOMINANTE = 0.60





def obtenir_cause_dominante(df_cdr, offer_code):

    echecs = df_cdr[(df_cdr["offer"] == offer_code) & (df_cdr["event_type"] == "transaction_failed")]

    if len(echecs) == 0:

        return None, 0.0

    compte = echecs["error_description"].value_counts()

    cause_principale = compte.index[0]

    part = compte.iloc[0] / len(echecs)

    return cause_principale, part





def generer_recommandation(ligne, cause, part_cause):

    taux = ligne["taux_succes_global"]
    variation = ligne["variation_ca_pct"]
    ca = ligne["ca_total"]
    volume = ligne["nb_souscriptions_total"]

    impact = (ligne["rang_ca"] + ligne["rang_volume"]) / 2 if "rang_ca" in ligne else 0.5

    if taux < SEUIL_TAUX_CRITIQUE:
        urgence = "critique" if impact > 0.5 else "elevee"
        if cause == CAUSE_CHRONIQUE and part_cause >= SEUIL_CAUSE_DOMINANTE:
            texte = (f"Offre fortement affectee par le bug technique chronique de la "
                     f"plateforme ({part_cause*100:.0f}% des echecs), pas un probleme isole. "
                     f"Taux de succes: {taux*100:.0f}%. Prioriser la resolution globale du bug "
                     f"plutot qu'une action specifique a cette offre.")
        elif part_cause >= SEUIL_CAUSE_DOMINANTE:
            texte = (f"Anomalie technique probablement specifique a cette offre "
                     f"(cause dominante: '{cause}', {part_cause*100:.0f}% des echecs). "
                     f"Taux de succes: {taux*100:.0f}%. A investiguer en priorite, "
                     f"defaut de configuration propre a cette offre suspecte.")
        else:
            texte = (f"Taux de succes critique ({taux*100:.0f}%) sans cause dominante claire "
                     f"(causes d'echec dispersees) - investigation approfondie necessaire.")
        priorite = f"technique_{urgence}"

    elif taux < SEUIL_TAUX_VIGILANCE:
        texte = (f"Fiabilite en zone de vigilance ({taux*100:.0f}%), en dessous du niveau "
                 f"sain (>70%) sans etre critique. A surveiller, pas d'action immediate requise.")
        priorite = "vigilance"

    elif variation > SEUIL_HAUSSE:
        texte = (f"Offre fiable (succes {taux*100:.0f}%) en forte croissance ({variation:+.1f}% "
                 f"de CA). Candidate a une promotion renforcee pour capitaliser sur la dynamique.")
        priorite = "opportunite"

    elif variation < SEUIL_BAISSE:
        if ca > 100000:
            texte = (f"Baisse marquee ({variation:+.1f}%) mais offre encore significative en CA "

                     f"({ca:.0f} DT) - repositionnement ou relance ciblee recommandee plutot "

                     f"qu'un retrait.")

        else:

            texte = (f"Baisse marquee ({variation:+.1f}%) sur une offre deja marginale "

                     f"({ca:.0f} DT de CA total) - envisager un retrait ou remplacement.")

        priorite = "marketing"



    else:

        texte = f"Performance stable (succes {taux*100:.0f}%, CA {ca:.0f} DT). Suivi normal."

        priorite = "neutre"



    return texte, priorite





def detecter_anomalies_statistiques(df):

    features = ["ca_total", "taux_succes_global", "nb_souscriptions_total", "variation_ca_pct"]

    x = df[features].copy()

    x = (x - x.mean()) / x.std()



    modele = IsolationForest(contamination=0.15, random_state=42)

    predictions = modele.fit_predict(x)

    scores = modele.decision_function(x)



    df["anomalie_statistique"] = (predictions == -1).astype(int)

    df["score_anomalie"] = scores.round(3)

    return df





def main():

    df = pd.read_csv(FICHIER_INDICE)

    df = detecter_anomalies_statistiques(df)

    df_cdr = pd.read_parquet(FICHIER_PARQUET)



    resultats = []

    for _, ligne in df.iterrows():

        cause, part = obtenir_cause_dominante(df_cdr, ligne["offer_code"])

        texte, priorite = generer_recommandation(ligne, cause, part)



        if ligne["anomalie_statistique"] == 1 and priorite in ("vigilance", "neutre"):

            texte += (" [Signal complementaire] Cette offre presente un profil statistiquement "

                      "atypique par rapport a l'ensemble des offres (detection multivariee), "

                      "meme si aucun seuil individuel n'est franchi - recommande une verification.")

            priorite = "anomalie_detectee"



        resultats.append({

            "offer_code": ligne["offer_code"],

            "offer_name": ligne["offer_name"],

            "indice_performance": ligne["indice_performance"],

            "anomalie_statistique": ligne["anomalie_statistique"],

            "cause_dominante": cause,

            "part_cause_dominante_pct": round(part * 100, 1),

            "priorite": priorite,

            "recommandation": texte,

        })



    resultats_df = pd.DataFrame(resultats)

    ordre = {

        "technique_critique": 0,

        "technique_elevee": 1,

        "anomalie_detectee": 2,

        "vigilance": 3,

        "opportunite": 4,

        "marketing": 5,

        "neutre": 6

    }

    resultats_df = resultats_df.sort_values(by="priorite", key=lambda x: x.map(ordre))

    resultats_df.to_csv(FICHIER_SORTIE, index=False)



    for priorite in ordre:

        sous = resultats_df[resultats_df["priorite"] == priorite]

        if len(sous) > 0:

            print(f"\n=== {priorite.upper()} ({len(sous)} offre(s)) ===")

            for _, l in sous.iterrows():

                print(f"\n{l['offer_code']} ({l['offer_name']}):")

                print(f"  {l['recommandation']}")



    print(f"\nFichier ecrit : {FICHIER_SORTIE}")





if __name__ == "__main__":

    main()
