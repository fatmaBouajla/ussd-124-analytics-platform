import glob

import hashlib

import hmac



import yaml

import pandas as pd



from config_secrets import CLE_HACHAGE_MSISDN



CDR_PATH_PATTERN = "/home/fatma/elk-ussd-orange/data-logs/orange/*.cdr"

OFFERS_CATALOGUE_PATH = "/home/fatma/elk-ussd-orange/logstash/dictionaries/offers_catalogue.yml"

ERROR_CODES_PATH = "/home/fatma/elk-ussd-orange/logstash/dictionaries/error_codes.yml"

OUTPUT_PARQUET_PATH = "/home/fatma/elk-ussd-orange/ml-pipeline/data/cdr_clean.parquet"



COLUMNS = ["timestamp", "service_class", "subscriber", "offer",

           "price", "result_code", "msg", "short_code", "parameters"]





def load_yaml_dict(path):

    with open(path, encoding="utf-8") as f:

        return yaml.safe_load(f)





def derive_event_type(msg):

    if msg == "Suivi conso":

        return "consultation"

    elif msg == "success":

        return "transaction_success"

    elif pd.notna(msg) and msg != "":

        return "transaction_failed"

    else:

        return "unknown"





def calculer_error_description(df, error_dict):
    error_description = pd.Series(pd.NA, index=df.index, dtype="object")
    masque_echec = df["event_type"] == "transaction_failed"

    masque_avec_code = masque_echec & df["result_code"].notna()
    code_str = df.loc[masque_avec_code, "result_code"].astype(int).astype(str)
    mappe = code_str.map(error_dict)
    fallback = "Code non documente (" + code_str + ")"
    error_description.loc[masque_avec_code] = mappe.fillna(fallback)

    masque_sans_code = masque_echec & df["result_code"].isna()
    error_description.loc[masque_sans_code] = df.loc[masque_sans_code, "msg"]

    return error_description


def hacher_msisdn(valeur, cle):
    if pd.isna(valeur) or valeur == "":
        return valeur
    return hmac.new(cle.encode(), str(valeur).encode(), hashlib.sha256).hexdigest()


def anonymiser_subscriber(df, cle):
    """
    Hachage HMAC-SHA256 avec cle secrete (pas un simple SHA-256) : sans la
    cle, impossible de reconstituer les MSISDN par force brute meme en
    connaissant l'espace complet des numeros possibles (voir discussion
    aout 2026). Seuls les identifiants uniques sont haches (pas ligne par
    ligne) pour rester rapide sur 4.4M lignes.
    """
    identifiants_uniques = df["subscriber"].unique()
    table_hachage = {v: hacher_msisdn(v, cle) for v in identifiants_uniques}
    return df["subscriber"].map(table_hachage)


def main():
    if not CLE_HACHAGE_MSISDN:
        raise ValueError(
            "CLE_HACHAGE_MSISDN n'est pas definie dans config_secrets.py. "
            "Generer une valeur avec : "
            "python3 -c \"import secrets; print(secrets.token_hex(32))\" "
            "et la coller dans config_secrets.py avant de lancer ce script."
        )

    files = sorted(glob.glob(CDR_PATH_PATTERN))
    print(f"Fichiers trouves : {len(files)}")

    df_list = []
    for f in files:
        df_list.append(pd.read_csv(
            f, sep=";", header=None, names=COLUMNS,
            dtype=str, keep_default_na=False
        ))
    df = pd.concat(df_list, ignore_index=True)
    print(f"Lignes lues (brut) : {len(df)}")

    before = len(df)
    df = df.drop_duplicates(keep="first")
    print(f"Doublons supprimes : {before - len(df)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["result_code"] = pd.to_numeric(df["result_code"], errors="coerce").astype("Int64")

    df["event_type"] = df["msg"].apply(derive_event_type)

    nb_avant_hachage = df["subscriber"].nunique()
    df["subscriber"] = anonymiser_subscriber(df, CLE_HACHAGE_MSISDN)
    nb_apres_hachage = df["subscriber"].nunique()

    offers_dict = load_yaml_dict(OFFERS_CATALOGUE_PATH)
    df["offer_name"] = df["offer"].map(offers_dict).fillna(df["offer"])
    df.loc[df["offer"] == "", "offer_name"] = pd.NA

    error_dict = load_yaml_dict(ERROR_CODES_PATH)
    df["error_description"] = calculer_error_description(df, error_dict)

    df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    print(f"Fichier ecrit : {OUTPUT_PARQUET_PATH}")
    print(f"Lignes finales : {len(df)}")
    print(f"\nAnonymisation subscriber : {nb_avant_hachage} identifiants uniques "
          f"avant hachage, {nb_apres_hachage} apres (doit etre egal - verification "
          f"qu'aucune collision n'a fusionne 2 clients differents).")
    print("\nRepartition event_type :")
    print(df["event_type"].value_counts())
    print("\nRepartition error_description (top 10) :")
    print(df["error_description"].value_counts().head(10))


if __name__ == "__main__":
    main()
