import pandas as pd

from sklearn.ensemble import RandomForestRegressor



FICHIER_TRAIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_train.csv"

FICHIER_TEST = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_test.csv"

FICHIER_INCIDENT = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_incident.csv"

FICHIER_RESULTAT = "/home/fatma/elk-ussd-orange/ml-pipeline/data/resultat_incident.csv"



SERIES = ["ca_reel", "nb_transactions", "taux_echec"]

TEMPOREL = ["heure_du_jour", "jour_semaine", "est_weekend"]



VARIANTE_GAGNANTE = {

    "ca_reel": "reduit",

    "nb_transactions": "complet",

    "taux_echec": "reduit",

}





def features_propres(serie):

    return [f"{serie}_moins_1h", f"{serie}_moins_2h", f"{serie}_moins_24h",

            f"{serie}_moyenne_3h", f"{serie}_moyenne_6h"]





def features_croisees_completes(serie):

    autres = [s for s in SERIES if s != serie]

    resultat = []

    for s in autres:

        resultat += features_propres(s)

    return resultat





def features_croisees_reduites(serie):

    autres = [s for s in SERIES if s != serie]

    resultat = []

    for s in autres:

        resultat += [f"{s}_moins_1h", f"{s}_moyenne_3h"]

    return resultat





def colonnes_finales(serie):

    croisees = (features_croisees_completes(serie)

                if VARIANTE_GAGNANTE[serie] == "complet"

                else features_croisees_reduites(serie))

    return features_propres(serie) + croisees + TEMPOREL





def main():

    train = pd.read_csv(FICHIER_TRAIN, parse_dates=["datetime"])

    test = pd.read_csv(FICHIER_TEST, parse_dates=["datetime"])

    train_complet = pd.concat([train, test], ignore_index=True)



    incident = pd.read_csv(FICHIER_INCIDENT, parse_dates=["datetime"])



    resultat = incident[["datetime"]].copy()



    for serie in SERIES:

        cible = f"{serie}_cible"

        colonnes = colonnes_finales(serie)



        modele = RandomForestRegressor(n_estimators=200, random_state=42)

        modele.fit(train_complet[colonnes], train_complet[cible])



        prediction = modele.predict(incident[colonnes])



        resultat[f"{serie}_reel"] = incident[cible].values

        resultat[f"{serie}_prevu"] = prediction.round(2)

        resultat[f"{serie}_ecart"] = (resultat[f"{serie}_reel"] - resultat[f"{serie}_prevu"]).round(2)



        importances = pd.Series(modele.feature_importances_, index=colonnes).sort_values(ascending=False)

        print(f"\n=== Importance des variables - {serie} ===")

        print(importances.head(5))



    resultat.to_csv(FICHIER_RESULTAT, index=False)



    print("\n=== Apercu resultat incident (reel vs prevu) ===")

    colonnes_affichage = ["datetime"] + [c for s in SERIES for c in

                                          [f"{s}_reel", f"{s}_prevu", f"{s}_ecart"]]

    print(resultat[colonnes_affichage].to_string(index=False))



    print(f"\nFichier ecrit : {FICHIER_RESULTAT}")





if __name__ == "__main__":

    main()
