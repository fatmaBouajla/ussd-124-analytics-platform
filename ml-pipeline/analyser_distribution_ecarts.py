import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from feature_utils import colonnes_finales

FICHIER_TRAIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_train.csv"
FICHIER_SORTIE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/distribution_ecarts_ca.csv"

SERIE_CIBLE = "ca_reel"


def main():
    train_complet = pd.read_csv(FICHIER_TRAIN, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)

    coupure = int(len(train_complet) * 0.8)
    train = train_complet.iloc[:coupure]
    validation = train_complet.iloc[coupure:]

    cible = f"{SERIE_CIBLE}_cible"
    colonnes = colonnes_finales(SERIE_CIBLE)

    modele = RandomForestRegressor(n_estimators=200, random_state=42)
    modele.fit(train[colonnes], train[cible])
    prediction = modele.predict(validation[colonnes])

    resultat = validation[["datetime"]].copy()
    resultat["ca_reel"] = validation[cible].values
    resultat["ca_prevu"] = prediction.round(2)

    resultat = resultat[resultat["ca_prevu"] > 1].copy()
    resultat["ecart_pct"] = (
        (resultat["ca_reel"] - resultat["ca_prevu"]) / resultat["ca_prevu"] * 100
    ).round(2)

    resultat.to_csv(FICHIER_SORTIE, index=False)

    print(f"Entrainement : {len(train)} lignes ({train['datetime'].min()} -> {train['datetime'].max()})")
    print(f"Validation    : {len(validation)} lignes ({validation['datetime'].min()} -> {validation['datetime'].max()})")
    print(resultat["ecart_pct"].describe())
    for p in [50, 75, 80, 85, 90, 95, 99]:
        valeur = resultat["ecart_pct"].abs().quantile(p / 100)
        print(f"  p{p} : {valeur:.1f}%")


if __name__ == "__main__":
    main()
