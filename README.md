# **Plateforme d’analyse des performances des offres USSD 124**

Projet de stage ingénieur **MLOps** : plateforme d’analyse des performances des offres USSD et d’aide à la décision, à partir des **CDR (Call Detail Records)** du service USSD 124.

Le système repère les incidents techniques tout seul, les distingue des comportements clients normaux, prévoit le CA à court terme, explique ses prédictions, et pousse le résultat dans des dashboards Kibana.

---

## **Architecture**
 Apache Airflow (orchestration)  déclenche et enchaîne toutes les étapes ci-dessous
```text
CDR bruts (.cdr)
    ↓
Nettoyage + anonymisation (HMAC-SHA256, irréversible)
    ↓
    ├── Indicateurs métier
    │       (classement des offres, tendances, causes d’échec)
    │
    ├── Prévision ML
    │       (CA, transactions, taux d’échec) + détection d’écart
    │           ↓
    │       Explicabilité SHAP, retraduite en langage métier
    │
    ├── Impact client (anonymisé)
    │
    └── Alertes + rapport automatique
            ↓
    Elasticsearch → Dashboards Kibana
```

---

## **Validation — protocole de test aveugle**


| Bloc                         | Rôle                                                                             |
| ---------------------------- | -------------------------------------------------------------------------------- |
| **Calibration**              | Entraînement du modèle et calcul des seuils, sur une période figée à l’avance    |
| **Validation rétrospective** | Un incident déjà connu pendant le développement |
| **Test aveugle**             | Période postérieure à la coupure, jamais vue par le modèle                       |

---

## **Choix du modèle**

Random Forest, XGBoost et Holt-Winters comparés sur la précision brute (**MAE/RMSE**) et l’utilité réelle (détecter une vraie baisse de CA, via **precision/recall** sur 5 fenêtres glissantes).

**Random Forest** gardé malgré un MAE légèrement moins bon : meilleure détection, et compatible SHAP nativement.

---

## **Stack**

### **Traitement & ML**

* Python
* pandas
* scikit-learn
* SHAP
* statsmodels
* XGBoost

### **Orchestration**

* Apache Airflow (pipeline complet automatisé, 17 tâches)

### **Stockage & supervision**

* Elasticsearch
* Logstash
* Kibana

### **Infrastructure**

* Docker
* Docker Compose
