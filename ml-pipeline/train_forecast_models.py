import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error



from feature_utils import (

    SERIES, TEMPOREL,

    features_propres, features_croisees_completes, features_croisees_reduites,

)



FICHIER_TRAIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_train.csv"

FICHIER_TEST = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_test.csv"

FICHIER_RESULTAT = "/home/fatma/elk-ussd-orange/ml-pipeline/data/comparaison_modeles.csv"





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
