import warnings

warnings.filterwarnings("ignore")



import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing

import xgboost as xgb



from feature_utils import SERIES, colonnes_finales



FICHIER_TRAIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_train.csv"

FICHIER_TEST = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_test.csv"

FICHIER_DATASET_30MIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/ml_dataset_30min.csv"

FICHIER_RESULTAT = "/home/fatma/elk-ussd-orange/ml-pipeline/data/comparaison_3_modeles.csv"





def evaluer_modele(modele, train, test, colonnes, cible):

    modele.fit(train[colonnes], train[cible])

    prediction = modele.predict(test[colonnes])

    mae = mean_absolute_error(test[cible], prediction)

    rmse = mean_squared_error(test[cible], prediction) ** 0.5

    r2 = r2_score(test[cible], prediction)
    return mae, rmse, r2


def evaluer_holt_winters(serie_train, serie_test):
    modele = ExponentialSmoothing(
        list(serie_train), trend=True, seasonal=48
    )
    resultat_ajuste = modele.fit(disp=False)

    predictions = []
    resultat_courant = resultat_ajuste
    for vraie_valeur in serie_test:
        pred = resultat_courant.forecast(1)[0]
        predictions.append(pred)
        resultat_courant = resultat_courant.append([vraie_valeur], refit=False)

    predictions = np.array(predictions)
    mae = mean_absolute_error(serie_test, predictions)
    rmse = mean_squared_error(serie_test, predictions) ** 0.5
    r2 = r2_score(serie_test, predictions)
    return mae, rmse, r2


def main():
    train = pd.read_csv(FICHIER_TRAIN, parse_dates=["datetime"])
    test = pd.read_csv(FICHIER_TEST, parse_dates=["datetime"])
    dataset_30min = pd.read_csv(FICHIER_DATASET_30MIN, parse_dates=["datetime"])

    date_train_min, date_train_max = train["datetime"].min(), train["datetime"].max()
    date_test_min, date_test_max = test["datetime"].min(), test["datetime"].max()

    resultats = []
    for serie in SERIES:
        cible = f"{serie}_cible"
        colonnes = colonnes_finales(serie)

        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        mae_rf, rmse_rf, r2_rf = evaluer_modele(rf, train, test, colonnes, cible)

        xgb_modele = xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
        mae_xgb, rmse_xgb, r2_xgb = evaluer_modele(xgb_modele, train, test, colonnes, cible)

        serie_train_brute = dataset_30min[
            (dataset_30min["datetime"] >= date_train_min) & (dataset_30min["datetime"] <= date_train_max)
        ][serie].values
        serie_test_brute = dataset_30min[
            (dataset_30min["datetime"] >= date_test_min) & (dataset_30min["datetime"] <= date_test_max)
        ][serie].values

        mae_hw, rmse_hw, r2_hw = evaluer_holt_winters(serie_train_brute, serie_test_brute)

        resultats.append({
           "cible": serie,
            "RF_MAE": round(mae_rf, 4), "RF_RMSE": round(rmse_rf, 4), "RF_R2": round(r2_rf, 4),
            "XGBoost_MAE": round(mae_xgb, 4), "XGBoost_RMSE": round(rmse_xgb, 4), "XGBoost_R2": round(r2_xgb, 4),
            "HoltWinters_MAE": round(mae_hw, 4), "HoltWinters_RMSE": round(rmse_hw, 4), "HoltWinters_R2": round(r2_hw, 4),
        })

    resultats_df = pd.DataFrame(resultats)
    print(resultats_df.to_string(index=False))
    resultats_df.to_csv(FICHIER_RESULTAT, index=False)
    print(f"\nFichier ecrit : {FICHIER_RESULTAT}")


if __name__ == "__main__":
    main()
