## 01 Entraînement XGBoost

#### Objectif

Ce document explique les aspects les plus importants de 01_train_xgboost.py.

#### Instructions pour l'exécution du code

Ouvrez 01_train_xgboost.py dans votre session CML et mettez à jour les variables DBNAME, STORAGE et CONNECTION_NAME aux lignes 60-62 comme indiqué par votre responsable HOL.

Ensuite, appuyez sur le bouton de lecture pour exécuter l'ensemble du script. Vous pourrez observer la sortie du code sur le côté droit de votre écran.

Accédez à l'onglet Expériences MLFLow et validez la création de l'expérience. Ouvrez l'exécution de l'expérience et familiarisez-vous avec les métadonnées de l'exécution de l'expérience. En bas de la page, ouvrez le dossier Artifacts et notez que les dépendances du modèle ont été suivies.

Ensuite, revenez à la session CML et modifiez le code :

* À la ligne 74, remplacez la ligne de code suivante :

```
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
```

par :

```
model = XGBClassifier(use_label_encoder=False, max_depth=4, eval_metric="logloss")
```

* À la ligne 95, ajoutez :

```
mlflow.log_param("max_depth", 4)
```

Exécutez le script. Ensuite, revenez à la page des Expériences MLFLow et validez la nouvelle exécution de l'expérience. Comparez les métadonnées de l'exécution de l'expérience avec celles de l'exécution précédente et constatez les modifications.

#### Points forts du code

* Ligne 41 : le package MLFlow est importé. MLFlow est installé par défaut dans les projets CML. Un plugin interne traduit les méthodes MLFlow en méthodes de l'API CML. Il n'est pas nécessaire d'installer ou de configurer MLFlow pour utiliser ses capacités de suivi.

* Lignes 72 - 100 : le code d'entraînement XGBoost est défini dans le contexte d'une exécution d'expérience MLFlow. Le code XGBoost reste inchangé. La méthode "mlflow.log_param()" est utilisée pour enregistrer les métriques du modèle. La méthode "mlflow.log_model()" est utilisée pour suivre les artefacts du modèle dans le dossier "artifacts".

* Ligne 120 : le client MLFlow est utilisé pour interagir avec les métadonnées de l'exécution de l'expérience. Vous pouvez utiliser le client pour rechercher des exécutions, comparer des résultats, et bien plus encore.

#### Résumé

Dans ce laboratoire, vous avez utilisé MLFlow pour suivre les exécutions d'expérience dans l'interface des expériences, accéder aux données des exécutions d'expérience de manière programmatique via le client MLFlow, et enregistrer les exécutions dans le registre MLFlow. Lorsqu'une exécution d'expérience est suivie, MLFlow stocke automatiquement les artefacts et les dépendances du modèle dans l'arrière-plan.

Le registre est un composant distinct de l'espace de travail et agit comme un environnement de staging pour déplacer éventuellement les modèles et les dépendances associées d'un espace de travail à un autre, par exemple dans un schéma DEV à QA à PRD.

MLFlow dans CML ne nécessite aucune installation ou configuration de la part des administrateurs ou des utilisateurs CML. Il est préinstallé par défaut dans chaque espace de travail CML. CML comprend un plugin spécial qui traduit les appels API MLFlow en routines de l'API CML v2. Vous en apprendrez davantage sur l'API CML v2 dans la section suivante.

#### Articles Connexes

* Pour en savoir plus sur MLFlow :
  * [Documentation MLFlow](https://mlflow.org/docs/latest/index.html)
  * [Enregistrement d'un modèle dans l'interface utilisateur MLFlow CML](https://docs.cloudera.com/machine-learning/1.5.4/models/topics/ml-registering-model-using-ui.html)
  * [Documentation MLFlow CML](https://docs.cloudera.com/machine-learning/cloud/experiments/topics/ml-experiments-v2.html)
  * [Ajustement des hyperparamètres avec les expériences sur CML](https://community.cloudera.com/t5/Community-Articles/Tuning-Hyperparameters-with-Experiments-feature-on-Cloudera/ta-p/368654)

* Pour en savoir plus sur XGBoost :
  * [Documentation XGboost](https://xgboost.readthedocs.io/en/stable/)
  * [XGBoost distribué avec PySpark dans CML](https://community.cloudera.com/t5/Community-Articles/Distributed-XGBoost-with-PySpark-in-Cloudera-Machine/ta-p/375810)
  * [XGBoost CML avec Dask AMP](https://github.com/cloudera/CML_AMP_Dask_on_CML)
