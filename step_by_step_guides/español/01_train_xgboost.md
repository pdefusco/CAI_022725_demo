## 01 Entrenamiento de XGBoost

#### Objetivo

Este documento explica los aspectos más importantes de 01_train_xgboost.py.

#### Instrucciones para la Ejecución del Código

Abre 01_train_xgboost.py en tu sesión de CML y actualiza las variables DBNAME, STORAGE y CONNECTION_NAME en las líneas 60-62 según las instrucciones de tu líder de HOL.

Luego, presiona el botón de reproducción para ejecutar todo el script. Podrás observar la salida del código en el lado derecho de tu pantalla.

Navega a la pestaña de Experimentos de MLFLow y valida la creación del experimento. Abre la Ejecución del Experimento y familiarízate con los metadatos de la Ejecución del Experimento. En la parte inferior de la página, abre la carpeta de Artefactos y nota que las dependencias del modelo han sido rastreadas.

Luego, regresa a la sesión de CML y modifica el código:

* En la línea 74, reemplaza la siguiente línea de código:

```
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
```

por:

```
model = XGBClassifier(use_label_encoder=False, max_depth=4, eval_metric="logloss")
```

* En la línea 95, agrega:

```
mlflow.log_param("max_depth", 4)
```

Ejecuta el script. Luego regresa a la página de Experimentos de MLFLow y valida la nueva Ejecución del Experimento. Compara los metadatos de la Ejecución del Experimento con la ejecución anterior y nota que han cambiado.

#### Aspectos Destacados del Código

* Línea 41: se importa el paquete MLFlow. MLFlow está instalado por defecto en los Proyectos CML. Un Plugin interno traduce los métodos de MLFlow a métodos de la API de CML. No es necesario instalar o configurar MLFlow para usar sus capacidades de Tracking.

* Líneas 72 - 100: el código de entrenamiento de XGBoost se define dentro del contexto de una Ejecución de Experimento de MLFlow. El código de XGBoost, de lo contrario, permanece sin cambios. El método "mlflow.log_param()" se utiliza para registrar métricas del modelo. El método "mlflow.log_model()" se utiliza para rastrear artefactos del modelo en la carpeta "artifacts".

* Línea 120: se utiliza el Cliente de MLFlow para interactuar con los metadatos de la Ejecución del Experimento. Puedes usar el Cliente para buscar entre ejecuciones, comparar resultados y mucho más.

#### Resumen

En este laboratorio utilizaste MLFlow para rastrear ejecuciones de experimentos en la interfaz de Experimentos, acceder a los datos de las ejecuciones de experimentos programáticamente a través del Cliente de MLFlow, y registrar Ejecuciones en el Registro de MLFlow. Cuando se rastrea una Ejecución de Experimento, MLFlow almacena automáticamente artefactos del modelo y dependencias en el backend.

El Registro es un componente separado del Espacio de Trabajo y actúa como un entorno de preparación para mover opcionalmente modelos y dependencias asociadas de un Espacio de Trabajo a otro, por ejemplo, de DEV a QA a PRD.

MLFlow en CML no requiere ninguna instalación ni configuraciones por parte de los Administradores o Usuarios de CML. Está preinstalado por defecto en cada Espacio de Trabajo de CML. CML incluye un Plugin especial que traduce las llamadas a la API de MLFlow a rutinas de la API v2 de CML. Aprenderás más sobre la API v2 de CML en la siguiente sección.

#### Artículos Relacionados

* Para aprender más sobre MLFlow:
  * [Documentación de MLFlow](https://mlflow.org/docs/latest/index.html)
  * [Registro de un Modelo en la Interfaz de MLFlow de CML](https://docs.cloudera.com/machine-learning/1.5.4/models/topics/ml-registering-model-using-ui.html)
  * [Documentación de MLFlow en CML](https://docs.cloudera.com/machine-learning/cloud/experiments/topics/ml-experiments-v2.html)
  * [Ajuste de Hiperparámetros con Experimentos en CML](https://community.cloudera.com/t5/Community-Articles/Tuning-Hyperparameters-with-Experiments-feature-on-Cloudera/ta-p/368654)

* Para aprender más sobre XGBoost:
  * [Documentación de XGBoost](https://xgboost.readthedocs.io/en/stable/)
  * [XGBoost Distribuido con PySpark en CML](https://community.cloudera.com/t5/Community-Articles/Distributed-XGBoost-with-PySpark-in-Cloudera-Machine/ta-p/375810)
  * [CML XGBoost con Dask AMP](https://github.com/cloudera/CML_AMP_Dask_on_CML)
