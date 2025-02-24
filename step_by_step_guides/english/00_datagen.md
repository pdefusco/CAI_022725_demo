## 00 Datagen

#### Objective

This document explains the most important aspects of 00_datagen.py.

#### Instructions for Code Execution

Open 00_datagen.py in your CML Session and update the CONNECTION_NAME variable at lines 162 as instructed by your HOL Lead.

Next, press the play button in order to run the whole script. Code output is available on the right side of your screen.

#### Code Highlights

* Line 50: the cml.data_v1 library is imported. This library allows you to take advantage of CML Data Connections in order to launch a Spark Session and connect to the Data Lake. The DataConnection is used at lines 103 - 109 within the "createSparkConnection" module.

* Lines 64 - 95: the "dataGen" module is used to create synthetic data for the classification use case. Observe the data attributes that are being created, and their respective types and value ranges.

* Lines 141 and 143: the PySark API for Apache Iceberg is used to create or append data to an Iceberg table format table from a PySpark dataframe.

#### Summary

In this lab you used CML Data Connections to preconfigure boiler plate code for data access to CDP and 3rd party data sources including Postgres, SQLServer, and even Snowflake. You can customize Data Connections to standardize and simplify Data Access configurations, as well as restrict access from 3rd party systems.

The PySpark Iceberg API allows you to create Iceberg tables from Spark DataFrames. It is similar and as simple as the standard PySpark API for interacting with Hive tables.

### Related Articles

* To learn more about CML Data Connections:
  * [Data Connections Article](https://community.cloudera.com/t5/Community-Articles/New-Feature-in-Cloudera-Machine-Learning-Data-Connections/ta-p/336775)
  * [Custom Data Connections](https://docs.cloudera.com/machine-learning/cloud/mlde/topics/ml-custom-data-conn-create.html)
  * [Custom Data Connections Article](https://community.cloudera.com/t5/Community-Articles/Using-Custom-Data-Connections-in-Cloudera-Machine-Learning/ta-p/379132)

* To learn more about Apache Iceberg:
  * [Apache Iceberg Documentation](https://iceberg.apache.org/docs/1.5.2/)
  * [Apache Iceberg with Spark](https://iceberg.apache.org/docs/1.5.2/spark-getting-started/)
  * [Apache Iceberg on Cloudera](https://www.cloudera.com/open-source/apache-iceberg.html)
  * [Apache Iceberg in CML](https://community.cloudera.com/t5/Community-Articles/Using-Cloudera-Machine-Learning-for-Datalake-and-Iceberg/ta-p/336133)
