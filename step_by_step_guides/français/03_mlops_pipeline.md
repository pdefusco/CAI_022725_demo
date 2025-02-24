## 03 Pipeline MLOps

#### Objectif

Ce document explique les aspects les plus importants de 03_newbatch.py, 04_train_xgboost.py et 05_api_redeployment.py.

#### Instructions pour l'exécution du code

Ouvrez 03_newbatch.py, 04_train_xgboost.py et 05_api_redeployment.py dans votre session CML. Familiarisez-vous avec le code et mettez à jour les variables DBNAME, STORAGE et CONNECTION_NAME comme indiqué par votre responsable HOL.

Ne lancez pas les scripts individuellement. Créez plutôt un travail CML pour chacun d'eux. Ne lancez pas encore les travaux.

Créez le travail "New Batch" avec les configurations suivantes :

```
Nom : New Batch Paul
Script : 03_newbatch.py
Éditeur : Workbench
Kernel : Python 3.9
Spark Add On : Spark 3.2 ou 3.3
Édition : Standard
Version : 2024.02
Planification : Manuelle
Profil de Ressource : 2 vCPU / 4 GiB / 0 GPU
```

Créez le travail "Retrain XGBoost" avec les configurations suivantes :

```
Nom : Retrain XGBoost Paul
Script : 04_train_xgboost.py
Éditeur : Workbench
Kernel : Python 3.9
Spark Add On : Spark 3.2 ou 3.3
Édition : Standard
Version : 2024.02
Planification : Dépendant de New Batch Paul
Profil de Ressource : 2 vCPU / 4 GiB / 0 GPU
```

Créez le travail "API Redeployment" avec les configurations suivantes :

```
Nom : API Redeployment Paul
Script : 05_api_redeployment.py
Éditeur : Workbench
Kernel : Python 3.9
Spark Add On : Spark 3.2 ou 3.3
Édition : Standard
Version : 2024.02
Planification : Dépendant de Retrain XGBoost Paul
Profil de Ressource : 2 vCPU / 4 GiB / 0 GPU
```

Une fois que vous avez créé les trois travaux, déclenchez manuellement le travail New Batch. Surveillez l'exécution dans l'onglet Historique des travaux et observez que, une fois terminé, le travail suivant dans le pipeline MLOps, Retrain XGBoost, est déclenché, et enfin le dernier travail, API Redeployment, est exécuté.

#### Points forts du code

* 03_newbatch.py est presque identique à 00_datagen.py.

* 04_train_xgboost.py est presque identique à "01_train_xgboost.py". Cependant, aux lignes 67-69, les métadonnées de l'instantané Iceberg sont stockées sous forme de variables. Ces métadonnées sont utilisées aux lignes 71-75 pour effectuer une lecture incrémentale, c'est-à-dire en ne chargeant que les données de la table Iceberg dans une plage horaire donnée. Les métadonnées sont ensuite enregistrées en tant que balises MLFlow pendant l'exécution de l'expérience.

* 05_api_redeployment.py comprend à la fois des méthodes de l'utilitaire mlops et du code pour exécuter le pipeline MLOps. Cela est également presque identique au code de "02_api_deployment.py".

#### Résumé

Dans ce laboratoire, vous avez utilisé les travaux CML en tandem avec l'API CMLv2, Apache Iceberg et MLFlow afin d'orchestrer un pipeline MLOps plus avancé. Avec seulement trois scripts et quelques lignes de code, vous avez mis en œuvre un processus CI/CD standardisé qui adhère aux meilleures pratiques MLOps, y compris la reproductibilité des données et des modèles, l'auditabilité et l'explicabilité. Vous avez réalisé cela en utilisant des composants intégrés et sans installations personnalisées.

* Dans le premier travail, un nouveau lot de données est ajouté à la table Iceberg Credit Card Transaction.

* Dans le deuxième travail, vous avez utilisé Iceberg Time Travel pour lire les données dans des plages horaires spécifiques - en d'autres termes, pour accéder uniquement au dernier lot de données - puis réentraîner le même classificateur XGBoost avec ces données.

* Enfin, dans le dernier travail, vous avez redéployé une nouvelle version de l'API Endpoint avec la dernière version du modèle.

#### Articles Connexes

* Pour en savoir plus sur les travaux CML :
  * [Créer un travail CML](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-job-c.html)
