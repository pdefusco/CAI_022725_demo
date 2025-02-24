## Setup

### Objective

This document provides instructions for setting up the HOL in your CML Workspace.

### Requirements

* CML Workspace on version xxx with instance types xxx
* CML MLFlow Registry
* CDP User with CML Admin rights and full setup in IDBroker Mappings and Ranger Hadoop SQL / RAZ-related Policies.
* CML Runtime Resource Profile of 2 vCPU and 4 or 8 GiB

### Setup Instructions

1. Deploy CML Project from Git Repository
2. Create a CML Session and Install Requirements

#### 1. Deploy CML Project from Git Repository

From inside the CML Workspace create a new project and enter the following parameters in the form:

```
Project Name: MLOps HOL <username>
Project Visibility: Private or Public
Initial Setup: Git -> https://github.com/pdefusco/CML_MLops_Banking_MLFlow.git
Runtimes:
  1. Remove all default runtimes.
  2. Select Advanced Options
  3. Select: Workbench Editor / Python 3.9 Kernel / Standard Edition / 2024.02 Version
```

![alt text](../../img/holbnk1.png)

![alt text](../../img/holbnk2.png)

#### 2. Create a CML Session and Install Requirements

Launch a CML Session with:

```
Editor: Workbench
Kernel: Python 3.9
Edition: Standard
Version: 2024.02
Enable Spark: Version 3.2 or 3.3
Resource Profile: 2 vCPU / 4 GiB Mem / 0 GPU
```

In the prompt on the right side enter the following command:

```
!pip3 install -r requirements.txt
```

![alt text](../../img/holbnk3.png)

![alt text](../../img/holbnk4.png)

Once all packages have been installed, proceed to instructions in 00_datagen.

![alt text](../../img/holbnk5.png)
