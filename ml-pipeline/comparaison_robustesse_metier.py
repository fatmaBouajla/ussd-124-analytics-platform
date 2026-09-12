import warnings

warnings.filterwarnings("ignore")



import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import (

    mean_absolute_error, mean_squared_error, r2_score,

    precision_score, recall_score, f1_score,

)

from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing



from feature_utils import colonnes_finales



FICHIER_TRAIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_train.csv"

FICHIER_TEST = "/home/fatma/elk-ussd-orange/ml-pipeline/data/forecast_test.csv"

FICHIER_DATASET_30MIN = "/home/fatma/elk-ussd-orange/ml-pipeline/data/ml_dataset_30min.csv"

FICHIER_SORTIE = "/home/fatma/elk-ussd-orange/ml-pipeline/data/robustesse_metier.csv"



SERIE = "ca_reel"

N_SPLITS = 5

SEUIL_BAISSE_DETECTION = -0.20  # baisse de 20%+ vs creneau precedent = evenement a detecter





def evaluer_holt_winters(serie_train, serie_test):

    modele = ExponentialSmoothing(list(serie_train), trend=True, seasonal=48)
    resultat_ajuste = modele.fit(disp=False)
    predictions = []
    resultat_courant = resultat_ajuste
    for vraie_valeur in serie_test:
        pred = resultat_courant.forecast(1)[0]
        predictions.append(pred)
        resultat_courant = resultat_courant.append([vraie_valeur], refit=False)
    return np.array(predictions)


def metriques_detection(actuel, precedent, predit):
    
    variation_reelle = (actuel - precedent) / precedent
    variation_predite = (predit - precedent) / precedent

    baisse_reelle = (variation_reelle <= SEUIL_BAISSE_DETECTION).astype(int)
    baisse_predite = (variation_predite <= SEUIL_BAISSE_DETECTION).astype(int)

    if baisse_reelle.sum() == 0:
        return None, None, None, int(baisse_reelle.sum())

    precision = precision_score(baisse_reelle, baisse_predite, zero_division=0)
    recall = recall_score(baisse_reelle, baisse_predite, zero_division=0)
    f1 = f1_score(baisse_reelle, baisse_predite, zero_division=0)
    return precision, recall, f1, int(baisse_reelle.sum())


def main():
    train = pd.read_csv(FICHIER_TRAIN, parse_dates=["datetime"])
    test = pd.read_csv(FICHIER_TEST, parse_dates=["datetime"])
    normal = pd.concat([train, test], ignore_index=True).sort_values("datetime").reset_index(drop=True)

    dataset_30min = pd.read_csv(FICHIER_DATASET_30MIN, parse_dates=["datetime"])

    cible = f"{SERIE}_cible"
    colonnes = colonnes_finales(SERIE)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    resultats = []
    for i, (idx_train, idx_test) in enumerate(tscv.split(normal), start=1):
        train_split = normal.iloc[idx_train]
        test_split = normal.iloc[idx_test]

        # --- Random Forest ---
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(train_split[colonnes], train_split[cible])
        pred_rf = rf.predict(test_split[colonnes])

        mae_rf = mean_absolute_error(test_split[cible], pred_rf)
        rmse_rf = mean_squared_error(test_split[cible], pred_rf) ** 0.5
        r2_rf = r2_score(test_split[cible], pred_rf)

        # --- Holt-Winters ---
        date_train_min, date_train_max = train_split["datetime"].min(), train_split["datetime"].max()
        date_test_min, date_test_max = test_split["datetime"].min(), test_split["datetime"].max()

        serie_train_brute = dataset_30min[
            (dataset_30min["datetime"] >= date_train_min) & (dataset_30min["datetime"] <= date_train_max)
        ][SERIE].values
        serie_test_brute = dataset_30min[
            (dataset_30min["datetime"] >= date_test_min) & (dataset_30min["datetime"] <= date_test_max)
        ][SERIE].values

        pred_hw = evaluer_holt_winters(serie_train_brute, serie_test_brute)
        mae_hw = mean_absolute_error(serie_test_brute, pred_hw)
        rmse_hw = mean_squared_error(serie_test_brute, pred_hw) ** 0.5
        r2_hw = r2_score(serie_test_brute, pred_hw)

        # --- Detection de baisse significative ---
        actuel = test_split[cible].values
        precedent = test_split[f"{SERIE}_moins_1h"].values  # dernier point connu avant la cible

        p_rf, r_rf, f1_rf, nb_evenements = metriques_detection(actuel, precedent, pred_rf)
        p_hw, r_hw, f1_hw, _ = metriques_detection(actuel, precedent, pred_hw)

        resultats.append({
            "split": i,
            "periode_test_debut": test_split["datetime"].min(),
            "periode_test_fin": test_split["datetime"].max(),
            "nb_baisses_significatives": nb_evenements,
            "RF_MAE": round(mae_rf, 2), "RF_RMSE": round(rmse_rf, 2), "RF_R2": round(r2_rf, 3),
            "RF_precision_detection": round(p_rf, 2) if p_rf is not None else None,
            "RF_recall_detection": round(r_rf, 2) if r_rf is not None else None,
           "RF_F1_detection": round(f1_rf, 2) if f1_rf is not None else None,
            "HW_MAE": round(mae_hw, 2), "HW_RMSE": round(rmse_hw, 2), "HW_R2": round(r2_hw, 3),
            "HW_precision_detection": round(p_hw, 2) if p_hw is not None else None,
            "HW_recall_detection": round(r_hw, 2) if r_hw is not None else None,
            "HW_F1_detection": round(f1_hw, 2) if f1_hw is not None else None,
        })

    resultats_df = pd.DataFrame(resultats)
    resultats_df.to_csv(FICHIER_SORTIE, index=False)

    print(resultats_df.to_string(index=False))

    print("\n=== Moyenne +/- ecart-type sur les 5 splits ===")
    for prefixe, nom in [("RF", "Random Forest"), ("HW", "Holt-Winters")]:
        mae_moy, mae_std = resultats_df[f"{prefixe}_MAE"].mean(), resultats_df[f"{prefixe}_MAE"].std()
        recall_moy = resultats_df[f"{prefixe}_recall_detection"].mean()
        precision_moy = resultats_df[f"{prefixe}_precision_detection"].mean()
        print(f"{nom} : MAE {mae_moy:.1f} (+/-{mae_std:.1f}) | "
              f"Recall detection baisse : {recall_moy:.2f} | "
              f"Precision detection baisse : {precision_moy:.2f}")

    print(f"\nFichier ecrit : {FICHIER_SORTIE}")


if __name__ == "__main__":
    main()
