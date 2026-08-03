"""

Fichier central de configuration des seuils metier du pipeline.



Regle : toute constante utilisee pour classer/filtrer/alerter doit vivre

ICI et nulle part ailleurs, pour eviter les divergences silencieuses entre

scripts (cf. bug rank_offers.py VOLUME_MIN=100 vs tendance_offres.py

VOLUME_MIN_TOTAL=1000, trouve et corrige en aout 2026).



Chaque valeur ci-dessous a ete discutee avec Fatma. Ne pas modifier une

valeur sans validation - preferer ouvrir la discussion avant de changer.

"""



# --- Fiabilite d'une offre ---

# Utilise par rank_offers.py, tendance_offres.py, indice_performance_offres.py.

# Une offre doit avoir au moins ce volume ET etre presente au moins ce

# nombre de jours pour etre consideree "fiable" (classement, tendance, indice).

VOLUME_MIN_FIABLE = 1000

JOURS_MIN_FIABLE = 10



# --- Tendance offre (tendance_offres.py) ---

SEUIL_TENDANCE = 0.10          # +/-10% = hausse/baisse, sinon "stable"

NB_JOURS_COMPARAISON = 5       # nb de jours compares en debut/fin de periode



# --- Recommandations par offre (generer_recommandations.py) ---
SEUIL_TAUX_CRITIQUE = 0.50
SEUIL_TAUX_VIGILANCE = 0.70
SEUIL_HAUSSE = 20.0
SEUIL_BAISSE = -30.0
SEUIL_CAUSE_DOMINANTE = 0.60
CONTAMINATION_ISOLATION_FOREST = 0.15

# --- Alerte CA (generer_rapport.py, recommandations_alertes.py) ---
# EN ATTENTE DE VALIDATION.
# A definir apres avoir lance analyser_distribution_ecarts.py et regarde
# la distribution reelle des ecarts CA reel/prevu en periode normale.
# Ne PAS mettre de valeur au hasard ici - laisser None tant que ce n'est
# pas discute et tranche avec Fatma.
SEUIL_ALERTE_CA_PCT = None  # ex. 20.0 une fois valide
