import pandas as pd
from elasticsearch import Elasticsearch, helpers

DATA_DIR = "/home/fatma/elk-ussd-orange/ml-pipeline/data"
ES_HOST = __import__("os").environ.get("ES_HOST", "http://localhost:9200")

es = Elasticsearch(ES_HOST)


def nettoyer_documents(df):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    df = df.astype(object).where(pd.notnull(df), None)
    return df.to_dict(orient="records")


def recreer_index(nom_index, mapping):
    if es.indices.exists(index=nom_index):
        es.indices.delete(index=nom_index)
        print(f"Index existant supprime : {nom_index}")
    es.indices.create(index=nom_index, mappings=mapping)
    print(f"Index cree : {nom_index}")


def creer_index_si_absent(nom_index, mapping):
    if not es.indices.exists(index=nom_index):
        es.indices.create(index=nom_index, mappings=mapping)
        print(f"Index cree : {nom_index}")
    else:
        print(f"Index existant conserve : {nom_index}")


def indexer(nom_index, documents, id_field=None):
    actions = []
    for doc in documents:
        action = {"_index": nom_index, "_source": doc}
        if id_field and doc.get(id_field) is not None:
            action["_id"] = f"{doc[id_field]}"
        actions.append(action)
    succes, erreurs = helpers.bulk(es, actions, raise_on_error=False)
    print(f"{nom_index} : {succes} documents indexes, {len(erreurs)} erreurs")
    if erreurs:
        print("Premieres erreurs :", erreurs[:3])


def indexer_upsert_partiel(nom_index, documents, id_field, champs_proteges):
    actions = []
    for doc in documents:
        doc_id = f"{doc[id_field]}_{doc.get('type_periode', '')}_{doc.get('type_alerte', '')}"
        doc_sans_champs_proteges = {k: v for k, v in doc.items() if k not in champs_proteges}
        actions.append({
            "_op_type": "update",
            "_index": nom_index,
            "_id": doc_id,
            "doc": doc_sans_champs_proteges,
            "upsert": doc,
        })
    succes, erreurs = helpers.bulk(es, actions, raise_on_error=False)
    print(f"{nom_index} : {succes} documents mis a jour (upsert), {len(erreurs)} erreurs")
    if erreurs:
        print("Premieres erreurs :", erreurs[:3])


def construire_index_offres():
    classement = pd.read_csv(f"{DATA_DIR}/offres_classement.csv")
    tendance = pd.read_csv(f"{DATA_DIR}/tendance_offres.csv")
    indice = pd.read_csv(f"{DATA_DIR}/indice_performance_offres.csv")
    constats = pd.read_csv(f"{DATA_DIR}/recommandations_offres.csv")

    df = classement.merge(
        tendance[["offer_code", "variation_volume_pct", "tendance_volume",
                  "variation_ca_pct", "tendance_ca"]],
        on="offer_code", how="left"
    )
    df = df.merge(indice[["offer_code", "indice_performance"]], on="offer_code", how="left")
    df = df.merge(
        constats[["offer_code", "priorite", "constat", "anomalie_statistique"]],
        on="offer_code", how="left"
    )

    mapping = {
        "properties": {
            "offer_code": {"type": "keyword"}, "offer_name": {"type": "keyword"},
            "ca_total": {"type": "float"}, "nb_souscriptions_total": {"type": "integer"},
            "nb_succes_total": {"type": "integer"}, "nb_echecs_total": {"type": "integer"},
            "nb_jours_presents": {"type": "integer"}, "taux_succes_global": {"type": "float"},
            "fiable": {"type": "boolean"}, "variation_volume_pct": {"type": "float"},
            "tendance_volume": {"type": "keyword"}, "variation_ca_pct": {"type": "float"},
            "tendance_ca": {"type": "keyword"}, "indice_performance": {"type": "float"},
            "priorite": {"type": "keyword"}, "constat": {"type": "text"},
            "anomalie_statistique": {"type": "integer"},
        }
    }
    recreer_index("ussd-offres", mapping)
    indexer("ussd-offres", nettoyer_documents(df), id_field="offer_code")


def construire_index_alertes():
    alertes_retro = pd.read_csv(f"{DATA_DIR}/recommandations_alertes.csv", parse_dates=["datetime"])
    alertes_retro["type_periode"] = "validation_retrospective"

    alertes_aveugle = pd.read_csv(f"{DATA_DIR}/recommandations_alertes_test_aveugle.csv", parse_dates=["datetime"])
    alertes_aveugle["type_periode"] = "test_aveugle"

    alertes = pd.concat([alertes_retro, alertes_aveugle], ignore_index=True)

    confiance_retro = pd.read_csv(f"{DATA_DIR}/confiance_incident.csv", parse_dates=["datetime"])
    confiance_aveugle = pd.read_csv(f"{DATA_DIR}/confiance_test_aveugle.csv", parse_dates=["datetime"])
    confiance = pd.concat([confiance_retro, confiance_aveugle], ignore_index=True)

    df = alertes.merge(confiance[["datetime", "ca_reel_hors_zone"]], on="datetime", how="left")
    df = df.rename(columns={"ca_reel_hors_zone": "hors_zone_confiance"})
    df["statut_alerte"] = "active"
    df["datetime_str"] = df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    mapping = {
        "properties": {
            "datetime": {"type": "date"}, "type_periode": {"type": "keyword"},
            "type_alerte": {"type": "keyword"}, "ca_ecart_pct": {"type": "float"},
            "ca_ecart_dt": {"type": "float"}, "nb_echecs_creneau": {"type": "integer"},
            "nb_clients_impactes": {"type": "integer"}, "nb_clients_reessai": {"type": "integer"},
            "cause_dominante": {"type": "keyword"}, "part_cause_dominante_pct": {"type": "float"},
            "hors_zone_confiance": {"type": "integer"},
            "taux_echec_reel": {"type": "float"}, "taux_echec_prevu": {"type": "float"},
            "taux_echec_ecart": {"type": "float"}, "statut_alerte": {"type": "keyword"},
        }
    }
    creer_index_si_absent("ussd-alertes", mapping)
    documents = nettoyer_documents(df.drop(columns=["datetime_str"]))
    for doc, doc_id_val in zip(documents, df["datetime_str"]):
        doc["_id_calcule"] = doc_id_val
    indexer_upsert_partiel("ussd-alertes", documents, id_field="_id_calcule", champs_proteges={"statut_alerte", "_id_calcule"})


def construire_index_clients():
    indicateurs = pd.read_csv(f"{DATA_DIR}/indicateurs_clients_30min.csv", parse_dates=["datetime"])
    indicateurs["type_document"] = "creneau_30min"

    bloques = pd.read_csv(f"{DATA_DIR}/clients_bloques.csv", parse_dates=["premiere_tentative", "derniere_tentative"])
    bloques["jour"] = bloques["derniere_tentative"].dt.date
    resume_jour = bloques.groupby("jour").agg(
        nb_clients_bloques=("subscriber", "count"),
        nb_echecs_moyen=("nb_echecs", "mean"), nb_echecs_max=("nb_echecs", "max"),
    ).reset_index()
    resume_jour["jour"] = pd.to_datetime(resume_jour["jour"])
    resume_jour["nb_echecs_moyen"] = resume_jour["nb_echecs_moyen"].round(1)
    resume_jour["type_document"] = "clients_bloques_par_jour"
    resume_jour = resume_jour.rename(columns={"jour": "datetime"})

    synthese = pd.DataFrame([{
        "datetime": "2026-07-09T11:50:00",
        "type_document": "synthese_globale_bloques",
        "nb_clients_bloques_total": len(bloques),
        "nb_echecs_moyen": round(bloques["nb_echecs"].mean(), 1),
        "nb_echecs_max": int(bloques["nb_echecs"].max()),
    }])

    mapping = {
        "properties": {
            "datetime": {"type": "date"}, "type_document": {"type": "keyword"},
            "nb_clients_impactes": {"type": "integer"}, "nb_clients_reessai": {"type": "integer"},
            "nb_clients_bloques": {"type": "integer"}, "nb_clients_bloques_total": {"type": "integer"},
            "nb_echecs_moyen": {"type": "float"}, "nb_echecs_max": {"type": "integer"},
        }
    }
    recreer_index("ussd-clients", mapping)
    indexer("ussd-clients", nettoyer_documents(indicateurs))
    indexer("ussd-clients", nettoyer_documents(resume_jour))
    indexer("ussd-clients", nettoyer_documents(synthese))


def construire_index_previsions():
    resultat_retro = pd.read_csv(f"{DATA_DIR}/resultat_incident.csv", parse_dates=["datetime"])
    resultat_retro["type_periode"] = "validation_retrospective"

    resultat_aveugle = pd.read_csv(f"{DATA_DIR}/resultat_test_aveugle.csv", parse_dates=["datetime"])
    resultat_aveugle["type_periode"] = "test_aveugle"

    resultat = pd.concat([resultat_retro, resultat_aveugle], ignore_index=True)

    confiance_retro = pd.read_csv(f"{DATA_DIR}/confiance_incident.csv", parse_dates=["datetime"])
    confiance_aveugle = pd.read_csv(f"{DATA_DIR}/confiance_test_aveugle.csv", parse_dates=["datetime"])
    confiance = pd.concat([confiance_retro, confiance_aveugle], ignore_index=True)

    df = resultat.merge(confiance, on="datetime", how="left")

    mapping = {
        "properties": {
            "datetime": {"type": "date"}, "type_periode": {"type": "keyword"},
            "ca_reel_reel": {"type": "float"}, "ca_reel_prevu": {"type": "float"},
            "ca_reel_ecart": {"type": "float"}, "nb_transactions_reel": {"type": "float"},
            "nb_transactions_prevu": {"type": "float"}, "nb_transactions_ecart": {"type": "float"},
            "taux_echec_reel": {"type": "float"}, "taux_echec_prevu": {"type": "float"},
            "taux_echec_ecart": {"type": "float"}, "ca_reel_hors_zone": {"type": "integer"},
            "nb_transactions_hors_zone": {"type": "integer"}, "taux_echec_hors_zone": {"type": "integer"},
        }
    }
    recreer_index("ussd-previsions", mapping)
    indexer("ussd-previsions", nettoyer_documents(df))


def construire_index_shap():
    lignes_importance = []
    for cible in ["ca_reel", "nb_transactions", "taux_echec"]:
        shap_df = pd.read_csv(f"{DATA_DIR}/shap_{cible}.csv")
        shap_df.columns = ["feature", "importance_shap"]
        shap_df["cible"] = cible
        shap_df["type_document"] = "importance_globale"
        lignes_importance.append(shap_df)
    importance_globale = pd.concat(lignes_importance, ignore_index=True)
    importance_globale["type_periode"] = "calibration"

    detail_retro = pd.read_csv(f"{DATA_DIR}/shap_incident_detail.csv", parse_dates=["datetime"])
    detail_retro["type_document"] = "explication_creneau"
    detail_retro["type_periode"] = "validation_retrospective"

    detail_aveugle = pd.read_csv(f"{DATA_DIR}/shap_test_aveugle_detail.csv", parse_dates=["datetime"])
    detail_aveugle["type_document"] = "explication_creneau"
    detail_aveugle["type_periode"] = "test_aveugle"

    mapping = {
        "properties": {
            "type_document": {"type": "keyword"}, "type_periode": {"type": "keyword"},
            "cible": {"type": "keyword"}, "feature": {"type": "keyword"}, "importance_shap": {"type": "float"},
            "datetime": {"type": "date"}, "ca_reel": {"type": "float"},
            "facteur_qui_a_le_plus_baisse_la_prevision": {"type": "keyword"}, "impact_dt": {"type": "float"},
            "facteur_qui_a_le_plus_augmente_la_prevision": {"type": "keyword"}, "impact_dt_positif": {"type": "float"},
        }
    }
    recreer_index("ussd-shap", mapping)
    indexer("ussd-shap", nettoyer_documents(importance_globale))
    indexer("ussd-shap", nettoyer_documents(detail_retro))
    indexer("ussd-shap", nettoyer_documents(detail_aveugle))


def construire_index_rapport():
    with open(f"{DATA_DIR}/rapport_incident.txt", encoding="utf-8") as f:
        texte = f.read()
    doc = pd.DataFrame([{"date_generation": "2026-07-09T11:50:00", "texte_rapport": texte}])
    mapping = {"properties": {"date_generation": {"type": "date"}, "texte_rapport": {"type": "text"}}}
    recreer_index("ussd-rapport", mapping)
    indexer("ussd-rapport", nettoyer_documents(doc))


def construire_index_offres_quotidien():
    offres = pd.read_csv(f"{DATA_DIR}/offres.csv", parse_dates=["date"])
    mapping = {
        "properties": {
            "date": {"type": "date"}, "offer_code": {"type": "keyword"}, "offer_name": {"type": "keyword"},
            "nb_souscriptions": {"type": "integer"}, "nb_succes": {"type": "integer"},
            "nb_echecs": {"type": "integer"}, "taux_succes": {"type": "float"}, "ca_offre": {"type": "float"},
        }
    }
    recreer_index("ussd-offres-quotidien", mapping)
    indexer("ussd-offres-quotidien", nettoyer_documents(offres))


def main():
    print("=== Indexation des resultats du pipeline ML vers Elasticsearch ===\n")
    construire_index_offres()
    construire_index_alertes()
    construire_index_clients()
    construire_index_previsions()
    construire_index_shap()
    construire_index_rapport()
    construire_index_offres_quotidien()
    print("\nTermine. 7 index : ussd-offres, ussd-alertes, ussd-clients, "
          "ussd-previsions, ussd-shap, ussd-rapport, ussd-offres-quotidien.")


if __name__ == "__main__":
    main()
