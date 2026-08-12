"""

Fichier central de configuration des seuils metier du pipeline.



Regle : toute constante utilisee pour classer/filtrer/alerter doit vivre

ICI et nulle part ailleurs, pour eviter les divergences silencieuses entre

scripts.

"""



# --- Fiabilite d'une offre ---

VOLUME_MIN_FIABLE = 1000

JOURS_MIN_FIABLE = 10



# --- Tendance offre (tendance_offres.py) ---

SEUIL_TENDANCE = 0.10

NB_JOURS_COMPARAISON = 5



# --- Recommandations par offre (generer_recommandations.py) ---

SEUIL_TAUX_CRITIQUE = 0.50

SEUIL_TAUX_VIGILANCE = 0.70

SEUIL_HAUSSE = 20.0

SEUIL_BAISSE = -30.0

SEUIL_CAUSE_DOMINANTE = 0.60

CONTAMINATION_ISOLATION_FOREST = 0.15



# --- Alerte CA (generer_rapport.py, recommandations_alertes.py) ---

# VALEUR PROVISOIRE : correspond au p90 observe sur la distribution des

# ecarts en periode normale (analyser_distribution_ecarts.py, aout 2026).

# A REMPLACER par un calcul dynamique (recalcule a chaque reentrainement,

# avec garde-fous min/max) une fois l'orchestration en place.

SEUIL_ALERTE_CA_PCT = 35.0

# --- Indicateurs clients (indicateurs_clients.py) ---
SEUIL_TENTATIVES_REESSAI = 2
SEUIL_TENTATIVES_BLOQUE = 3
