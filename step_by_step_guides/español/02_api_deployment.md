## 02 Despliegue de API

#### Objetivo

Este documento explica los aspectos más importantes de 02_api_deployment.py.

#### Instrucciones para la Ejecución del Código

Abre 02_api_deployment.py en tu sesión de CML. Familiarízate con el código. Luego, abre mlops.py y también familiarízate con el código.

A continuación, regresa a 02_api_deployment.py y presiona el botón de reproducción para ejecutar todo el script. Podrás observar la salida del código en el lado derecho de tu pantalla.

#### Aspectos Destacados del Código

* Línea 46: se importa la clase "ModelDeployment" desde el utilitario "mlops". Este utilitario se ha colocado en la carpeta "/home/cdsw".

* Línea 49: se instancia el cliente de la API de CML. La API te proporciona más de 100 métodos en Python para ejecutar acciones como crear proyectos, lanzar trabajos, y mucho más. En este ejemplo, la API se usa para "list_projects()".

* Línea 62: el Cliente de la API se pasa como argumento a la instancia de ModelDeployment. El utilitario mlops.py incluye algunos métodos que extienden y sobrescriben métodos de la API. Típicamente, los Ingenieros de Machine Learning de CML crean Interfaces en Python para construir pipelines de MLOps personalizados según lo requiera su caso de uso.

* Línea 68: la última Ejecución de Experimento de MLFlow se usa para registrar el Modelo en el Registro de MLFlow de CML.

* Líneas 74, 78, 81: el Modelo registrado se usa para crear un nuevo Despliegue de Modelo en CML. El Modelo primero se crea, luego se construye, y finalmente se despliega.

#### Resumen

En este laboratorio usaste la APIv2 de CML para ejecutar acciones programáticamente dentro de los Espacios de Trabajo de CML. Puedes usar la API con simples comandos curl CLI o con el Wrapper de Python, que es una biblioteca preinstalada en cada Runtime de Cloudera CML. La API puede ser utilizada tanto desde sistemas externos al Espacio de Trabajo de CML como dentro del mismo Espacio de Trabajo de CML.

Los Científicos de Datos de CML aprovechan la API para construir Pipelines de MLOps. En este laboratorio usaste una interfaz en Python simple para sobrescribir métodos de la API con el fin de construir un pipeline de MLOps estandarizado para registrar una Ejecución de Experimento en el Registro de MLFlow y luego desplegar un Endpoint de API para servir el modelo.

#### Artículos Relacionados

* Para aprender más sobre los Despliegues de Modelos en CML:
  * [Artículo sobre Despliegue de Modelos en CML con MLFlow y APIv2](https://community.cloudera.com/t5/Community-Articles/CML-Model-Deployment-with-MLFlow-and-APIv2/ta-p/385656)

* Para aprender más sobre la API de CML:
  * [CML APIv2](https://docs.cloudera.com/machine-learning/cloud/api/topics/ml-api-v2.html)
  * [Referencia de la API REST de CML](https://docs.cloudera.com/machine-learning/cloud/rest-api-reference/index.html)
  * [CML API v2 AMP](https://github.com/cloudera/CML_AMP_APIv2)
