## 03 Pipeline de MLOps

#### Objetivo

Este documento explica los aspectos más importantes de 03_newbatch.py, 04_train_xgboost.py y 05_api_redeployment.py.

#### Instrucciones para la Ejecución del Código

Abre 03_newbatch.py, 04_train_xgboost.py y 05_api_redeployment.py en tu sesión de CML. Familiarízate con el código y actualiza las variables DBNAME, STORAGE y CONNECTION_NAME según las instrucciones de tu líder de HOL.

No ejecutes los scripts individualmente. Crea un Trabajo de CML para cada uno en su lugar. No ejecutes los trabajos aún.

Crea el Trabajo "New Batch" con las siguientes configuraciones:

```
Nombre: New Batch Paul
Script: 03_newbatch.py
Editor: Workbench
Kernel: Python 3.9
Complemento Spark: Spark 3.2 o 3.3
Edición: Standard
Versión: 2024.02
Programación: Manual
Perfil de Recursos: 2 vCPU / 4 GiB / 0 GPU
```

Crea el Trabajo "Retrain XGBoost" con las siguientes configuraciones:

```
Nombre: Retrain XGBoost Paul
Script: 04_train_xgboost.py
Editor: Workbench
Kernel: Python 3.9
Complemento Spark: Spark 3.2 o 3.3
Edición: Standard
Versión: 2024.02
Programación: Dependiente de New Batch Paul
Perfil de Recursos: 2 vCPU / 4 GiB / 0 GPU
```

Crea el Trabajo "API Redeployment" con las siguientes configuraciones:

```
Nombre: API Redeployment Paul
Script: 05_api_redeployment.py
Editor: Workbench
Kernel: Python 3.9
Complemento Spark: Spark 3.2 o 3.3
Edición: Standard
Versión: 2024.02
Programación: Dependiente de Retrain XGBoost Paul
Perfil de Recursos: 2 vCPU / 4 GiB / 0 GPU
```

Una vez que hayas creado los tres trabajos, activa manualmente el trabajo New Batch. Monitorea la ejecución en la pestaña de Historial de Trabajos y observa que, una vez que se complete, se activa el siguiente trabajo en el pipeline de MLOps, Retrain XGBoost, y finalmente se ejecuta el último trabajo, API Redeployment.

#### Aspectos Destacados del Código

* 03_newbatch.py es en su mayor parte idéntico a 00_datagen.py.

* 04_train_xgboost.py es casi idéntico a "01_train_xgboost.py". Sin embargo, en las líneas 67-69, se almacenan los metadatos del Snapshot de Iceberg como variables. Estos metadatos se utilizan en las líneas 71-75 para realizar una Lectura Incremental, es decir, cargar solo los datos de la tabla Iceberg dentro de un límite de tiempo de inicio y fin. Los metadatos se guardan luego como Etiquetas de MLFlow durante la ejecución de la Ejecución del Experimento.

* 05_api_redeployment.py incluye tanto métodos del utilitario mlops como código para ejecutar el pipeline de MLOps. Esto también es casi idéntico al código en "02_api_deployment.py".

#### Resumen

En este laboratorio usaste Trabajos de CML junto con la APIv2 de CML, Apache Iceberg y MLFlow para orquestar un pipeline de MLOps más avanzado. Con solo tres scripts y unas pocas líneas de código, has implementado un Proceso CI/CD estandarizado que adhiere a las Mejores Prácticas de MLOps, incluyendo reproducibilidad de datos y modelos, auditabilidad y explicabilidad. Hiciste esto aprovechando los componentes integrados y sin instalaciones personalizadas.

* En el primer trabajo, se añade un nuevo lote de datos a la tabla de Transacciones con Tarjeta de Crédito de Iceberg.

* En el segundo trabajo, usaste el Time Travel de Iceberg para leer datos dentro de límites de tiempo específicos - en otras palabras, para acceder solo al último lote de datos - y luego volver a entrenar el mismo Clasificador XGBoost con estos datos.

* Finalmente, en el último trabajo, se volvió a desplegar una nueva versión del Endpoint de API con la versión más reciente del modelo.

#### Artículos Relacionados

* Para aprender más sobre Trabajos de CML:
  * [Creación de un Job en CML](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-job-c.html)
