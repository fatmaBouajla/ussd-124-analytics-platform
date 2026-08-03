import pandas as pd

from sklearn.ensemble import RandomForestRegressor



from feature_utils import colonnes_finales



FICHIER_TRAIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_train.csv"

FICHIER_TEST = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_test.csv"

FICHIER_SORTIE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/distribution_ecarts_ca.csv"



SERIE_CIBLE = "ca_reel"





def main():

    train = pd.read_csv(FICHIER_TRAIN, parse_dates=["datetime"])

    test = pd.read_csv(FICHIER_TEST, parse_dates=["datetime"])



    cible = f"{SERIE_CIBLE}_cible"

    colonnes = colonnes_finales(SERIE_CIBLE)



    modele = RandomForestRegressor(n_estimators=200, random_state=42)

    modele.fit(train[colonnes], train[cible])

    prediction = modele.predict(test[colonnes])



    resultat = test[["datetime"]].copy()

    resultat["ca_reel"] = test[cible].values

    resultat["ca_prevu"] = prediction.round(2)



    # evite une division par 0 ou quasi-0 qui exploserait le %

    resultat = resultat[resultat["ca_prevu"] > 1].copy()
    resultat["ecart_pct"] = (
        (resultat["ca_reel"] - resultat["ca_prevu"]) / resultat["ca_prevu"] * 100
    ).round(2)

    resultat.to_csv(FICHIER_SORTIE, index=False)

    print("=== Distribution des ecarts CA reel vs CA prevu (periode normale, test set) ===")
    print(resultat["ecart_pct"].describe())

    print("\nPercentiles de |ecart_pct| (utile pour choisir un seuil d'alerte) :")
    for p in [50, 75, 80, 85, 90, 95, 99]:
        valeur = resultat["ecart_pct"].abs().quantile(p / 100)
        print(f"  p{p} : {valeur:.1f}%")

    print(f"\nFichier ecrit : {FICHIER_SORTIE}")
    print(f"Nombre de creneaux analyses : {len(resultat)}")


if __name__ == "__main__":
    main()
