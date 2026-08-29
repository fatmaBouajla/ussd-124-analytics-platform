from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

SCRIPTS_DIR = "/home/fatma/elk-ussd-orange/ml-pipeline"
PYTHON_CMD = f"cd {SCRIPTS_DIR} && python3 "

default_args = {
    "owner": "fatma",
    "retries": 0,
}

with DAG(
    dag_id="ussd_pipeline_complet",
    description="Pipeline complet USSD 124 : ingestion, offres, ML, alertes, rapport, indexation",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2026, 8, 1),
    catchup=False,
    tags=["ussd", "production"],
) as dag:

    extract_cdr = BashOperator(
        task_id="extract_cdr",
        bash_command=f"{PYTHON_CMD}extract_cdr.py",
    )

    build_kpi_global = BashOperator(
        task_id="build_kpi_global",
        bash_command=f"{PYTHON_CMD}build_kpi_global.py",
    )

    build_offres = BashOperator(
        task_id="build_offres",
        bash_command=f"{PYTHON_CMD}build_offres.py",
    )

    build_ml_dataset_30min = BashOperator(
        task_id="build_ml_dataset_30min",
        bash_command=f"{PYTHON_CMD}build_ml_dataset_30min.py",
    )

    rank_offers = BashOperator(
        task_id="rank_offers",
        bash_command=f"{PYTHON_CMD}rank_offers.py",
    )

    tendance_offres = BashOperator(
        task_id="tendance_offres",
        bash_command=f"{PYTHON_CMD}tendance_offres.py",
    )

    indice_performance_offres = BashOperator(
        task_id="indice_performance_offres",
        bash_command=f"{PYTHON_CMD}indice_performance_offres.py",
    )

    generer_recommandations = BashOperator(
        task_id="generer_recommandations",
        bash_command=f"{PYTHON_CMD}generer_recommandations.py",
    )

    indicateurs_clients = BashOperator(
        task_id="indicateurs_clients",
        bash_command=f"{PYTHON_CMD}indicateurs_clients.py",
    )

    build_forecast_dataset = BashOperator(
        task_id="build_forecast_dataset",
        bash_command=f"{PYTHON_CMD}build_forecast_dataset.py",
    )

    train_final_models = BashOperator(
        task_id="train_final_models",
        bash_command=f"{PYTHON_CMD}train_final_models.py",
    )

    calculer_confiance = BashOperator(
        task_id="calculer_confiance",
        bash_command=f"{PYTHON_CMD}calculer_confiance.py",
    )

    shap_analysis = BashOperator(
        task_id="shap_analysis",
        bash_command=f"{PYTHON_CMD}shap_analysis.py",
    )

    shap_incident = BashOperator(
        task_id="shap_incident",
        bash_command=f"{PYTHON_CMD}shap_incident.py",
    )

    recommandations_alertes = BashOperator(
        task_id="recommandations_alertes",
        bash_command=f"{PYTHON_CMD}recommandations_alertes.py",
    )

    generer_rapport = BashOperator(
        task_id="generer_rapport",
        bash_command=f"{PYTHON_CMD}generer_rapport.py",
    )

    indexer_resultats = BashOperator(
        task_id="indexer_resultats",
        bash_command=f"{PYTHON_CMD}indexer_resultats.py",
        env={"ES_HOST": "http://es-ussd:9200"},
    )

    extract_cdr >> [build_kpi_global, build_offres, build_ml_dataset_30min, indicateurs_clients]

    build_offres >> [rank_offers, tendance_offres]
    [rank_offers, tendance_offres] >> indice_performance_offres
    indice_performance_offres >> generer_recommandations

    build_ml_dataset_30min >> build_forecast_dataset
    build_forecast_dataset >> [train_final_models, calculer_confiance, shap_analysis]
    shap_analysis >> shap_incident

    [train_final_models, calculer_confiance, indicateurs_clients] >> recommandations_alertes
    [recommandations_alertes, shap_analysis, rank_offers, indicateurs_clients] >> generer_rapport

    [generer_recommandations, recommandations_alertes, generer_rapport,
     indicateurs_clients, calculer_confiance, shap_incident, build_offres] >> indexer_resultats
