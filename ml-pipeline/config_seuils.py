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

# EN ATTENTE - calcul dynamique prevu lors de la mise en place de

# l'orchestration (recalcule a chaque cycle, avec garde-fous min/max).

SEUIL_ALERTE_CA_PCT = None



# --- Indicateurs clients (indicateurs_clients.py) ---

# Nb minimum d'echecs d'un meme client DANS LE MEME CRENEAU DE 30 MIN

# pour le compter comme "reessaie plusieurs fois" (signal d'acharnement
# immediat, typique d'un incident technique en cours).
SEUIL_TENTATIVES_REESSAI = 2

# Nb minimum d'echecs d'un meme client SUR TOUTE LA PERIODE, avec 0 succes,
# pour le classer "client bloque" (evite de classer comme bloque un client
# qui a juste tente une fois sans jamais retenter).
SEUIL_TENTATIVES_BLOQUE = 3
