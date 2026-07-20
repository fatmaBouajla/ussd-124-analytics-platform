import glob
import yaml
import pandas as pd

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


def main():
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

    offers_dict = load_yaml_dict(OFFERS_CATALOGUE_PATH)
    df["offer_name"] = df["offer"].map(offers_dict).fillna(df["offer"])
    df.loc[df["offer"] == "", "offer_name"] = pd.NA

    error_dict = load_yaml_dict(ERROR_CODES_PATH)

    def map_error(row):
        if row["event_type"] != "transaction_failed":
            return pd.NA
        code = row["result_code"]
        if pd.notna(code):
            code_str = str(int(code))
            if code_str in error_dict:
                return error_dict[code_str]
            return f"Code non documente ({code_str})"
        return row["msg"]

    df["error_description"] = df.apply(map_error, axis=1)

    df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    print(f"Fichier ecrit : {OUTPUT_PARQUET_PATH}")
    print(f"Lignes finales : {len(df)}")
    print("\nRepartition event_type :")
    print(df["event_type"].value_counts())
    print("\nRepartition error_description (top 10) :")
    print(df["error_description"].value_counts().head(10))


if __name__ == "__main__":
    main()
