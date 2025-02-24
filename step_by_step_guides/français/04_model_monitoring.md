## 04 Surveillance des Modèles

#### Objectif

Ce document explique les aspects les plus importants de 06_model_simulation.py et 07_model_monitoring.py.

#### Instructions pour l'exécution du code

Ouvrez 06_model_simulation.py et 07_model_monitoring.py dans votre session CML. Familiarisez-vous avec le code et mettez à jour les variables DBNAME, STORAGE et CONNECTION_NAME comme indiqué par votre responsable HOL.

Exécutez 06_model_simulation.py, puis quittez immédiatement la session pour accéder au Déploiement du Modèle CML. Ouvrez l'onglet Monitoring et regardez le tableau de bord de surveillance se mettre à jour en temps réel au fur et à mesure que les requêtes sont reçues par le point de terminaison du modèle.

Ensuite, exécutez 07_model_monitoring.py. Explorez les diagrammes de surveillance du modèle sur le côté droit de la page. Quelle est la performance de votre modèle ?

#### Points Clés du Code

06_model_simulation.py génère plus de données synthétiques et utilise le SDK CDSW pour interagir avec les points de terminaison déployés. Les données sont soumises au modèle avec la vérité de base pour simuler une vague de requêtes vers le point de terminaison.

* Ligne 41 : le SDK cdsw est importé. Il n'est pas nécessaire de l'installer.
* Lignes 91-129 : une méthode pour soumettre des lots de requêtes au point de terminaison du Déploiement du Modèle spécifié.
* Lignes 134-154 : une méthode pour soumettre la vérité de base au point de terminaison du Déploiement du Modèle spécifié.

07_model_monitoring.py vous montre comment vous pouvez surveiller la performance du modèle de manière programmatique. CML dispose d'une base de données Postgres appelée "Model Metrics Store" qui stocke automatiquement les métadonnées pour chaque requête par modèle déployé. Dans ce script, le SDK cdsw est utilisé à nouveau pour lire les métadonnées du modèle et accéder au Model Metrics Store.

* Lignes 66-67 : l'ApiUtility est instancié pour obtenir les métadonnées du modèle. Le code source de l'utilitaire est situé dans le dossier src. De la même manière que l'utilitaire "mlops" utilisé dans 02_api_deployment.py, cet utilitaire vous permet de créer des méthodes personnalisées pour obtenir les métadonnées du modèle selon les besoins de votre cas d'utilisation.

* Ligne 76 : cdsw.read_metrics() est utilisé pour lire les requêtes de modèle suivies depuis le Model Metrics Store.

* Seaborn et Pandas sont utilisés tout au long du reste du script pour créer les graphiques de performance du modèle.

#### Résumé

Dans ce laboratoire, vous avez utilisé le SDK cdsw pour accéder aux requêtes de prédiction et à la vérité de base stockées dans le Model Metrics Store de CML afin de surveiller la performance du modèle (précision).

Vous pouvez utiliser les mêmes outils en conjonction avec les Jobs CML planifiés pour créer des systèmes de surveillance continue et, de manière similaire aux laboratoires précédents, déployer de manière programmatique de nouvelles versions du modèle lorsque les critères de performance ne sont pas satisfaits.

#### Articles Connexes

* Pour en savoir plus sur le suivi des métriques des modèles dans CML :
  * [Documentation sur les Métriques des Modèles](https://docs.cloudera.com/machine-learning/cloud/model-metrics/topics/ml-enabling-model-metrics.html)
  * [Gouvernance des Modèles](https://docs.cloudera.com/machine-learning/cloud/model-governance/topics/ml-enabling-model-governance.html)

* Pour en savoir plus sur les Déploiements et la Surveillance des Modèles CML :
  * [Création et Déploiement d'un Modèle](https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-creating-and-deploying-a-model.html)
  * [Surveillance Continue des Modèles CML AMP](https://github.com/cloudera/CML_AMP_Continuous_Model_Monitoring)
