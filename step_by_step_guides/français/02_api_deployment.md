## 02 Déploiement de l'API

#### Objectif

Ce document explique les aspects les plus importants de 02_api_deployment.py.

#### Instructions pour l'exécution du code

Ouvrez 02_api_deployment.py dans votre session CML. Familiarisez-vous avec le code. Ensuite, ouvrez mlops.py et familiarisez-vous également avec le code.

Ensuite, revenez à 02_api_deployment.py et appuyez sur le bouton de lecture pour exécuter l'ensemble du script. Vous pourrez observer la sortie du code sur le côté droit de votre écran.

#### Points clés du code

* Ligne 46 : la classe "ModelDeployment" est importée depuis l'utilitaire "mlops". Cet utilitaire a été placé dans le dossier "/home/cdsw".

* Ligne 49 : le client API CML est instancié. L'API vous fournit plus de 100 méthodes Python pour exécuter des actions telles que la création de projets, le lancement de tâches, et bien plus encore. Dans cet exemple, l'API est utilisée pour "list_projects()".

* Ligne 62 : le client API est passé en argument à l'instance ModelDeployment. L'utilitaire mlops.py inclut quelques méthodes qui étendent et remplacent les méthodes API. En général, les ingénieurs en apprentissage automatique de CML créent des interfaces Python pour construire des pipelines MLOps personnalisés en fonction de leurs besoins.

* Ligne 68 : le dernier Exécution d'Expérience MLFlow est utilisé pour enregistrer le modèle dans le registre MLFlow de CML.

* Lignes 74, 78, 81 : le modèle enregistré est utilisé pour créer un nouveau Déploiement de Modèle CML. Le modèle est d'abord créé, puis construit, et enfin déployé.

#### Résumé

Dans ce laboratoire, vous avez utilisé CML APIv2 pour exécuter des actions de manière programmatique au sein des Espaces de Travail CML. Vous pouvez utiliser l'API avec des commandes curl CLI simples ou avec le Wrapper Python, qui est une bibliothèque préinstallée dans chaque Runtime Cloudera CML. L'API peut être utilisée à la fois par des systèmes tiers externes à l'Espace de Travail CML et au sein de l'Espace de Travail CML.

Les scientifiques des données CML exploitent l'API pour construire des pipelines MLOps. Dans ce laboratoire, vous avez utilisé une interface Python simple pour remplacer les méthodes API afin de construire un pipeline MLOps standardisé pour enregistrer une Exécution d'Expérience dans le registre MLFlow, puis déployer un point de terminaison API pour le service du modèle.

#### Articles Connexes

* Pour en savoir plus sur les Déploiements de Modèles CML :
  * [Déploiement de Modèle CML avec MLFlow et APIv2 Article](https://community.cloudera.com/t5/Community-Articles/CML-Model-Deployment-with-MLFlow-and-APIv2/ta-p/385656)

* Pour en savoir plus sur l'API CML :
  * [CML APIv2](https://docs.cloudera.com/machine-learning/cloud/api/topics/ml-api-v2.html)
  * [Référence API REST CML](https://docs.cloudera.com/machine-learning/cloud/rest-api-reference/index.html)
  * [CML API v2 AMP](https://github.com/cloudera/CML_AMP_APIv2)
