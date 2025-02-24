## 00 Datagen

#### Objetivo

Este documento explica los aspectos más importantes de 00_datagen.py.

#### Instrucciones para la Ejecución del Código

Abre 00_datagen.py en tu sesión de CML y actualiza las variable CONNECTION_NAME en las línea 162 según las instrucciones de tu líder de HOL.

Luego, presiona el botón de reproducción para ejecutar todo el script. La salida del código está disponible en el lado derecho de tu pantalla.

#### Aspectos Destacados del Código

* Línea 50: se importa la biblioteca cml.data_v1. Esta biblioteca te permite aprovechar las Conexiones de Datos de CML para iniciar una Sesión de Spark y conectarte al Data Lake. La DataConnection se utiliza en las líneas 103-109 dentro del módulo "createSparkConnection".

* Líneas 64-95: se utiliza el módulo "dataGen" para crear datos sintéticos para el caso de uso de clasificación. Observa los atributos de datos que se están creando, así como sus respectivos tipos y rangos de valores.

* Líneas 141 y 143: se utiliza la API de PySpark para Apache Iceberg para crear o agregar datos a una tabla en formato Iceberg a partir de un dataframe de PySpark.

#### Resumen

En este laboratorio utilizaste las Conexiones de Datos de CML para preconfigurar código base para el acceso a datos en CDP y fuentes de datos de terceros, incluyendo Postgres, SQLServer e incluso Snowflake. Puedes personalizar las Conexiones de Datos para estandarizar y simplificar las configuraciones de Acceso a Datos, así como restringir el acceso desde sistemas de terceros.

La API de PySpark Iceberg te permite crear tablas Iceberg a partir de DataFrames de Spark. Es similar y tan simple como la API estándar de PySpark para interactuar con tablas de Hive.

### Artículos Relacionados

* Para aprender más sobre las Conexiones de Datos de CML:
  * [Artículo sobre Conexiones de Datos](https://community.cloudera.com/t5/Community-Articles/New-Feature-in-Cloudera-Machine-Learning-Data-Connections/ta-p/336775)
  * [Conexiones de Datos Personalizadas](https://docs.cloudera.com/machine-learning/cloud/mlde/topics/ml-custom-data-conn-create.html)
  * [Artículo sobre Conexiones de Datos Personalizadas](https://community.cloudera.com/t5/Community-Articles/Using-Custom-Data-Connections-in-Cloudera-Machine-Learning/ta-p/379132)

* Para aprender más sobre Apache Iceberg:
  * [Documentación de Apache Iceberg](https://iceberg.apache.org/docs/1.5.2/)
  * [Apache Iceberg con Spark](https://iceberg.apache.org/docs/1.5.2/spark-getting-started/)
  * [Apache Iceberg en Cloudera](https://www.cloudera.com/open-source/apache-iceberg.html)
  * [Apache Iceberg en CML](https://community.cloudera.com/t5/Community-Articles/Using-Cloudera-Machine-Learning-for-Datalake-and-Iceberg/ta-p/336133)
