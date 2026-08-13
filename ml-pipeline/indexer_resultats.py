import pandas as pd
from elasticsearch import Elasticsearch, helpers

DATA_DIR = "/home/fatma/elk-ussd-orange/ml-pipeline/data"

# Elasticsearch tourne sur la VM via le port 9200
ES_HOST = "http://localhost:9200"

es = Elasticsearch(ES_HOST)


def nettoyer_documents(df):
    """Remplace NaN/NaT par None (JSON-compatible) et convertit les dates
    en chaines ISO avant indexation. Cast en object AVANT le remplacement,
    sinon pandas reconvertit automatiquement None -> NaN dans les colonnes
    numeriques (bug rencontre en aout 2026)."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    df = df.astype(object).where(pd.notnull(df), None)
    return df.to_dict(orient="records")


def recreer_index(nom_index, mapping):
    if es.indices.exists(index=nom_index):
        es.indices.delete(index=nom_index)
        print(f"Index existant supprimé : {nom_index}")

    es.indices.create(index=nom_index, mappings=mapping)
    print(f"Index créé : {nom_index}")


def indexer(nom_index, documents, id_field=None):
    actions = []

    for doc in documents:
        action = {
            "_index": nom_index,
            "_source": doc
        }

        if id_field and doc.get(id_field) is not None:
            action["_id"] = f"{doc[id_field]}"

        actions.append(action)

    succes, erreurs = helpers.bulk(
        es,
        actions,
        raise_on_error=False
    )

    print(f"{nom_index} : {succes} documents indexés, {len(erreurs)} erreurs")

    if erreurs:
        print("Premières erreurs :", erreurs[:3])


# Index des offres : classement, tendances, score et recommandations
def construire_index_offres():
    classement = pd.read_csv(f"{DATA_DIR}/offres_classement.csv")
    tendance = pd.read_csv(f"{DATA_DIR}/tendance_offres.csv")
    indice = pd.read_csv(f"{DATA_DIR}/indice_performance_offres.csv")
    constats = pd.read_csv(f"{DATA_DIR}/recommandations_offres.csv")

    df = classement.merge(
        tendance[
            [
                "offer_code",
                "variation_volume_pct",
                "tendance_volume",
                "variation_ca_pct",
                "tendance_ca"
            ]
        ],
        on="offer_code",
        how="left"
    )

    df = df.merge(
        indice[
            [
                "offer_code",
                "indice_performance"
            ]
        ],
        on="offer_code",
        how="left"
    )

    df = df.merge(
        constats[
            [
                "offer_code",
                "priorite",
                "constat",
                "anomalie_statistique"
            ]
        ],
        on="offer_code",
        how="left"
    )

    mapping = {
        "properties": {
            "offer_code": {"type": "keyword"},
            "offer_name": {"type": "keyword"},
            "ca_total": {"type": "float"},
            "nb_souscriptions_total": {"type": "integer"},
            "nb_succes_total": {"type": "integer"},
            "nb_echecs_total": {"type": "integer"},
            "nb_jours_presents": {"type": "integer"},
            "taux_succes_global": {"type": "float"},
            "fiable": {"type": "boolean"},
            "variation_volume_pct": {"type": "float"},
            "tendance_volume": {"type": "keyword"},
            "variation_ca_pct": {"type": "float"},
            "tendance_ca": {"type": "keyword"},
            "indice_performance": {"type": "float"},
            "priorite": {"type": "keyword"},
            "constat": {"type": "text"},
            "anomalie_statistique": {"type": "integer"}
        }
    }

    recreer_index("ussd-offres", mapping)
    indexer(
        "ussd-offres",
        nettoyer_documents(df),
        id_field="offer_code"
    )


# Index des alertes et des incidents détectés
def construire_index_alertes():
    alertes = pd.read_csv(
        f"{DATA_DIR}/recommandations_alertes.csv",
        parse_dates=["datetime"]
    )

    confiance = pd.read_csv(
        f"{DATA_DIR}/confiance_incident.csv",
        parse_dates=["datetime"]
    )

    df = alertes.merge(
        confiance[
            [
                "datetime",
                "ca_reel_hors_zone"
            ]
        ],
        on="datetime",
        how="left"
    )

    df = df.rename(
        columns={
            "ca_reel_hors_zone": "hors_zone_confiance"
        }
    )

    mapping = {
        "properties": {
            "datetime": {"type": "date"},
            "ca_ecart_pct": {"type": "float"},
            "ca_ecart_dt": {"type": "float"},
            "nb_echecs_creneau": {"type": "integer"},
            "nb_clients_impactes": {"type": "integer"},
            "nb_clients_reessai": {"type": "integer"},
            "cause_dominante": {"type": "keyword"},
            "part_cause_dominante_pct": {"type": "float"},
            "hors_zone_confiance": {"type": "integer"}
        }
    }

    recreer_index("ussd-alertes", mapping)
    indexer(
        "ussd-alertes",
        nettoyer_documents(df)
    )


# Index des indicateurs clients et des clients bloqués
def construire_index_clients():
    indicateurs = pd.read_csv(
        f"{DATA_DIR}/indicateurs_clients_30min.csv",
        parse_dates=["datetime"]
    )

    indicateurs["type_document"] = "creneau_30min"

    bloques = pd.read_csv(
        f"{DATA_DIR}/clients_bloques.csv",
        parse_dates=[
            "premiere_tentative",
            "derniere_tentative"
        ]
    )

    bloques["jour"] = bloques["derniere_tentative"].dt.date

    resume_jour = (
        bloques
        .groupby("jour")
        .agg(
            nb_clients_bloques=("subscriber", "count"),
            nb_echecs_moyen=("nb_echecs", "mean"),
            nb_echecs_max=("nb_echecs", "max")
        )
        .reset_index()
    )

    resume_jour["jour"] = pd.to_datetime(resume_jour["jour"])
    resume_jour["nb_echecs_moyen"] = resume_jour["nb_echecs_moyen"].round(1)
    resume_jour["type_document"] = "clients_bloques_par_jour"

    resume_jour = resume_jour.rename(
        columns={"jour": "datetime"}
    )

    synthese = pd.DataFrame([
        {
            "datetime": pd.Timestamp.now().strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            "type_document": "synthese_globale_bloques",
            "nb_clients_bloques_total": len(bloques),
            "nb_echecs_moyen": round(
                bloques["nb_echecs"].mean(),
                1
            ),
            "nb_echecs_max": int(
                bloques["nb_echecs"].max()
            )
        }
    ])

    mapping = {
        "properties": {
            "datetime": {"type": "date"},
            "type_document": {"type": "keyword"},
            "nb_clients_impactes": {"type": "integer"},
            "nb_clients_reessai": {"type": "integer"},
            "nb_clients_bloques": {"type": "integer"},
            "nb_clients_bloques_total": {"type": "integer"},
            "nb_echecs_moyen": {"type": "float"},
            "nb_echecs_max": {"type": "integer"}
        }
    }

    recreer_index("ussd-clients", mapping)

    indexer(
        "ussd-clients",
        nettoyer_documents(indicateurs)
    )

    indexer(
        "ussd-clients",
        nettoyer_documents(resume_jour)
    )

    indexer(
        "ussd-clients",
        nettoyer_documents(synthese)
    )

    print(
        f"\nNote : les {len(bloques)} clients bloqués "
        f"restent dans {DATA_DIR}/clients_bloques.csv "
        f"et ne sont pas indexés dans Kibana."
    )


# Comparaison des prédictions du modèle avec les valeurs réelles
def construire_index_previsions():
    resultat = pd.read_csv(
        f"{DATA_DIR}/resultat_incident.csv",
        parse_dates=["datetime"]
    )

    confiance = pd.read_csv(
        f"{DATA_DIR}/confiance_incident.csv",
        parse_dates=["datetime"]
    )

    df = resultat.merge(
        confiance,
        on="datetime",
        how="left"
    )

    mapping = {
        "properties": {
            "datetime": {"type": "date"},
            "ca_reel_reel": {"type": "float"},
            "ca_reel_prevu": {"type": "float"},
            "ca_reel_ecart": {"type": "float"},
            "nb_transactions_reel": {"type": "float"},
            "nb_transactions_prevu": {"type": "float"},
            "nb_transactions_ecart": {"type": "float"},
            "taux_echec_reel": {"type": "float"},
            "taux_echec_prevu": {"type": "float"},
            "taux_echec_ecart": {"type": "float"},
            "ca_reel_hors_zone": {"type": "integer"},
            "nb_transactions_hors_zone": {"type": "integer"},
            "taux_echec_hors_zone": {"type": "integer"}
        }
    }

    recreer_index("ussd-previsions", mapping)

    indexer(
        "ussd-previsions",
        nettoyer_documents(df)
    )


def main():
    print(
        "=== Indexation des résultats du pipeline ML "
        "vers Elasticsearch ===\n"
    )

    construire_index_offres()
    construire_index_alertes()
    construire_index_clients()
    construire_index_previsions()

    print(
        "\nTerminé. Vérifier dans Kibana que les 4 index apparaissent : "
        "ussd-offres, ussd-alertes, ussd-clients, ussd-previsions."
    )


if __name__ == "__main__":
    main()
