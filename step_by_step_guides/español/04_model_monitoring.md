## 04 Monitoreo de Modelos

#### Objetivo

Este documento explica los aspectos más importantes de 06_model_simulation.py y 07_model_monitoring.py.

#### Instrucciones para la Ejecución del Código

Abre 06_model_simulation.py y 07_model_monitoring.py en tu sesión de CML. Familiarízate con el código y actualiza las variables DBNAME, STORAGE y CONNECTION_NAME según las instrucciones de tu líder de HOL.

Ejecuta 06_model_simulation.py y, de inmediato, sal de la sesión para ir al Despliegue de Modelos de CML. Abre la pestaña de Monitoreo y observa cómo el panel de monitoreo se actualiza en tiempo real a medida que el endpoint del modelo recibe solicitudes.

A continuación, ejecuta 07_model_monitoring.py. Explora los diagramas de monitoreo del modelo en el lado derecho de la página. ¿Cómo está funcionando tu modelo?

#### Aspectos Destacados del Código

06_model_simulation.py crea más datos sintéticos y utiliza el SDK de CDSW para interactuar con los endpoints desplegados. Los datos se envían al modelo junto con la verdad básica para simular una ola de solicitudes al endpoint.

* Línea 41: se importa el SDK de cdsw. No es necesario instalarlo.
* Líneas 91-129: un método para enviar lotes de solicitudes al endpoint del Despliegue de Modelo especificado.
* Líneas 134-154: un método para enviar la verdad básica al endpoint del Despliegue de Modelo especificado.

07_model_monitoring.py te muestra cómo puedes monitorear el rendimiento del modelo de manera programática. CML cuenta con una base de datos Postgres llamada "Model Metrics Store" que almacena automáticamente metadatos para cada solicitud por modelo desplegado. En este script, el SDK de cdsw se utiliza nuevamente para leer metadatos del modelo y acceder al Model Metrics Store.

* Líneas 66-67: se instancia el ApiUtility para obtener metadatos del modelo. El código fuente del utilitario se encuentra en la carpeta src. De manera similar al utilitario "mlops" que usaste en 02_api_deployment.py, este utilitario te permite construir métodos personalizados para obtener metadatos del modelo según lo requiera tu caso de uso.

* Línea 76: se usa cdsw.read_metrics() para leer las solicitudes de modelo rastreadas desde el Model Metrics Store.

* Seaborn y Pandas se usan en el resto del script para crear los gráficos de rendimiento del modelo.

#### Resumen

En este laboratorio usaste el SDK de cdsw para acceder a las solicitudes de predicciones y a la verdad básica respaldadas en el Model Metrics Store de CML con el fin de monitorear el rendimiento del modelo (Precisión).

Puedes aprovechar las mismas herramientas en conjunto con Trabajos de CML programados para crear sistemas de monitoreo continuo y, de manera similar a los laboratorios anteriores, desplegar programáticamente nuevas versiones del modelo cuando los criterios de rendimiento no se cumplan.

#### Artículos Relacionados

* Para aprender más sobre el Seguimiento de Métricas de Modelos en CML:
  * [Documentación sobre Métricas de Modelos](https://docs.cloudera.com/machine-learning/cloud/model-metrics/topics/ml-enabling-model-metrics.html)
  * [Gobernanza de Modelos](https://docs.cloudera.com/machine-learning/cloud/model-governance/topics/ml-enabling-model-governance.html)

* Para aprender más sobre Despliegues y Monitoreo de Modelos en CML:
  * [Creación y Despliegue de un Modelo](https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-creating-and-deploying-a-model.html)
  * [Monitoreo Continuo de Modelos en CML AMP](https://github.com/cloudera/CML_AMP_Continuous_Model_Monitoring)
