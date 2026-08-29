import yaml

FICHIER_CLASSIFICATION = "/home/fatma/elk-ussd-orange/ml-pipeline/classification_causes.yml"

STATUT_PAR_DEFAUT = "a_verifier"


def charger_classification():
    with open(FICHIER_CLASSIFICATION, encoding="utf-8") as f:
        return yaml.safe_load(f)


def statut_cause(cause, classification=None):
    if classification is None:
        classification = charger_classification()
    entree = classification.get(cause)
    if entree is None:
        return STATUT_PAR_DEFAUT
    return entree.get("statut", STATUT_PAR_DEFAUT)


def est_technique(cause, classification=None):
    return statut_cause(cause, classification) == "technique_ponctuelle"


def interpretation_cause(cause, classification=None):
    if classification is None:
        classification = charger_classification()
    entree = classification.get(cause)
    if entree is None:
        return "Cause non repertoriee dans la classification de reference - statut incertain, a valider."
    return entree.get("interpretation", "")
