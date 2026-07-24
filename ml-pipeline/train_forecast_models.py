import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error



FICHIER_TRAIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_train.csv"

FICHIER_TEST = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_test.csv"

FICHIER_RESULTAT = "/home/fatma/elk-ussd-orange/ml-pipeline/data/comparaison_modeles.csv"



SERIES = ["ca_reel", "nb_transactions", "taux_echec"]

TEMPOREL = ["heure_du_jour", "jour_semaine", "est_weekend", "position_periode"]





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


def entrainer_evaluer(train, test, colonnes, cible):
    modele = RandomForestRegressor(n_estimators=200, random_state=42)
    modele.fit(train[colonnes], train[cible])
    prediction = modele.predict(test[colonnes])
    mae = mean_absolute_error(test[cible], prediction)
    rmse = mean_squared_error(test[cible], prediction) ** 0.5
    return mae, rmse


def main():
    train = pd.read_csv(FICHIER_TRAIN, parse_dates=["datetime"])
    test = pd.read_csv(FICHIER_TEST, parse_dates=["datetime"])

    resultats = []
    for serie in SERIES:
        cible = f"{serie}_cible"

        colonnes_completes = features_propres(serie) + features_croisees_completes(serie) + TEMPOREL
        mae_complet, rmse_complet = entrainer_evaluer(train, test, colonnes_completes, cible)

        colonnes_reduites = features_propres(serie) + features_croisees_reduites(serie) + TEMPOREL
        mae_reduit, rmse_reduit = entrainer_evaluer(train, test, colonnes_reduites, cible)

        resultats.append({
            "cible": serie,
            "nb_features_complet": len(colonnes_completes),
            "MAE_complet": round(mae_complet, 4),
            "RMSE_complet": round(rmse_complet, 4),
            "nb_features_reduit": len(colonnes_reduites),
            "MAE_reduit": round(mae_reduit, 4),
            "RMSE_reduit": round(rmse_reduit, 4),
        })

    resultats_df = pd.DataFrame(resultats)
    print(resultats_df.to_string(index=False))
    resultats_df.to_csv(FICHIER_RESULTAT, index=False)


if __name__ == "__main__":
    main()
