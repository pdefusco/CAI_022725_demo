## Configuration

### Objectif

Ce document fournit des instructions pour configurer le HOL dans votre espace de travail CML.

### Exigences

* Espace de travail CML en version xxx avec types d'instances xxx
* Registre MLFlow CML
* Utilisateur CDP avec droits d'administration CML et configuration complète dans les Mappings IDBroker et les politiques Hadoop SQL / RAZ de Ranger.
* Profil de ressources d'exécution CML de 2 vCPU et 4 ou 8 GiB

### Instructions de configuration

1. Déployez le projet CML depuis le dépôt Git
2. Créez une session CML et installez les prérequis

#### 1. Déployer le projet CML depuis le dépôt Git

Depuis l'espace de travail CML, créez un nouveau projet et entrez les paramètres suivants dans le formulaire :

```
Nom du projet : MLOps HOL <username>
Visibilité du projet : Privé ou Public
Configuration initiale : Git -> https://github.com/pdefusco/CML_MLops_Banking_MLFlow.git
Runtimes :
  1. Supprimez tous les runtimes par défaut.
  2. Sélectionnez Options Avancées
  3. Sélectionnez : Éditeur Workbench / Kernel Python 3.9 / Édition Standard / Version 2024.02
```

![alt text](../../img/holbnk1.png)

![alt text](../../img/holbnk2.png)

#### 2. Créer une session CML et installer les prérequis

Lancez une session CML avec :

```
Éditeur : Workbench
Kernel : Python 3.9
Édition : Standard
Version : 2024.02
Activer Spark : Version 3.2 ou 3.3
Profil de ressources : 2 vCPU / 4 GiB Mémoire / 0 GPU
```

Dans l'invite sur le côté droit, entrez la commande suivante :

```
!pip3 install -r requirements.txt
```

![alt text](../../img/holbnk3.png)

![alt text](../../img/holbnk4.png)

Une fois tous les packages installés, passez aux instructions dans 00_datagen.

![alt text](../../img/holbnk5.png)
