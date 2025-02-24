## 00 Datagen

#### Objectif

Ce document explique les aspects les plus importants de 00_datagen.py.

#### Instructions pour l'exécution du code

Ouvrez 00_datagen.py dans votre session CML et mettez à jour la variable CONNECTION_NAME à ligne 162 comme indiqué par votre responsable HOL.

Ensuite, appuyez sur le bouton de lecture pour exécuter l'ensemble du script. La sortie du code est disponible sur le côté droit de votre écran.

#### Points forts du code

* Ligne 50 : la bibliothèque cml.data_v1 est importée. Cette bibliothèque vous permet de tirer parti des connexions de données CML pour lancer une session Spark et vous connecter au Data Lake. La DataConnection est utilisée aux lignes 103 - 109 dans le module "createSparkConnection".

* Lignes 64 - 95 : le module "dataGen" est utilisé pour créer des données synthétiques pour le cas d'utilisation de classification. Observez les attributs de données qui sont créés, ainsi que leurs types et plages de valeurs respectifs.

* Lignes 141 et 143 : l'API PySpark pour Apache Iceberg est utilisée pour créer ou ajouter des données à une table au format Iceberg à partir d'un DataFrame PySpark.

#### Résumé

Dans ce laboratoire, vous avez utilisé les connexions de données CML pour préconfigurer le code de base pour l'accès aux données de CDP et de sources de données tierces, y compris Postgres, SQLServer et même Snowflake. Vous pouvez personnaliser les connexions de données pour standardiser et simplifier les configurations d'accès aux données, ainsi que restreindre l'accès aux systèmes tiers.

L'API PySpark Iceberg vous permet de créer des tables Iceberg à partir de DataFrames Spark. Elle est similaire et aussi simple que l'API PySpark standard pour interagir avec les tables Hive.

### Articles Connexes

* Pour en savoir plus sur les connexions de données CML :
  * [Article sur les Connexions de Données](https://community.cloudera.com/t5/Community-Articles/New-Feature-in-Cloudera-Machine-Learning-Data-Connections/ta-p/336775)
  * [Connexions de Données Personnalisées](https://docs.cloudera.com/machine-learning/cloud/mlde/topics/ml-custom-data-conn-create.html)
  * [Article sur les Connexions de Données Personnalisées](https://community.cloudera.com/t5/Community-Articles/Using-Custom-Data-Connections-in-Cloudera-Machine-Learning/ta-p/379132)

* Pour en savoir plus sur Apache Iceberg :
  * [Documentation d'Apache Iceberg](https://iceberg.apache.org/docs/1.5.2/)
  * [Apache Iceberg avec Spark](https://iceberg.apache.org/docs/1.5.2/spark-getting-started/)
  * [Apache Iceberg sur Cloudera](https://www.cloudera.com/open-source/apache-iceberg.html)
  * [Apache Iceberg dans CML](https://community.cloudera.com/t5/Community-Articles/Using-Cloudera-Machine-Learning-for-Datalake-and-Iceberg/ta-p/336133)
