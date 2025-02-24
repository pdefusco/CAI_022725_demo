## 00 Datagen

#### Obiettivo

Questo documento spiega gli aspetti più importanti di 00_datagen.py.

#### Istruzioni per l'esecuzione del codice

Apri 00_datagen.py nella tua sessione CML e aggiorna la variabile CONNECTION_NAME alla riga 162 come indicato dal tuo responsabile HOL.

Successivamente, premi il pulsante di riproduzione per eseguire l'intero script. L'output del codice è disponibile sul lato destro dello schermo.

#### Punti salienti del codice

* Riga 50: viene importata la libreria cml.data_v1. Questa libreria ti consente di sfruttare le Connessioni Dati CML per avviare una Sessione Spark e connetterti al Data Lake. La DataConnection viene utilizzata alle righe 103 - 109 nel modulo "createSparkConnection".

* Righe 64 - 95: il modulo "dataGen" viene utilizzato per creare dati sintetici per il caso d'uso della classificazione. Osserva gli attributi dei dati che vengono creati, insieme ai loro tipi e intervalli di valori rispettivi.

* Righe 141 e 143: l'API PySpark per Apache Iceberg viene utilizzata per creare o aggiungere dati a una tabella nel formato Iceberg a partire da un DataFrame PySpark.

#### Riepilogo

In questo laboratorio, hai utilizzato le Connessioni Dati CML per preconfigurare il codice di base per l'accesso ai dati di CDP e alle fonti di dati di terze parti, inclusi Postgres, SQLServer e anche Snowflake. Puoi personalizzare le Connessioni Dati per standardizzare e semplificare le configurazioni di accesso ai dati, nonché per limitare l'accesso ai sistemi di terze parti.

L'API PySpark Iceberg ti consente di creare tabelle Iceberg a partire da DataFrames Spark. È simile e altrettanto semplice quanto l'API PySpark standard per interagire con le tabelle Hive.

### Articoli Correlati

* Per saperne di più sulle Connessioni Dati CML:
  * [Articolo sulle Connessioni Dati](https://community.cloudera.com/t5/Community-Articles/New-Feature-in-Cloudera-Machine-Learning-Data-Connections/ta-p/336775)
  * [Connessioni Dati Personalizzate](https://docs.cloudera.com/machine-learning/cloud/mlde/topics/ml-custom-data-conn-create.html)
  * [Articolo sulle Connessioni Dati Personalizzate](https://community.cloudera.com/t5/Community-Articles/Using-Custom-Data-Connections-in-Cloudera-Machine-Learning/ta-p/379132)

* Per saperne di più su Apache Iceberg:
  * [Documentazione di Apache Iceberg](https://iceberg.apache.org/docs/1.5.2/)
  * [Apache Iceberg con Spark](https://iceberg.apache.org/docs/1.5.2/spark-getting-started/)
  * [Apache Iceberg su Cloudera](https://www.cloudera.com/open-source/apache-iceberg.html)
  * [Apache Iceberg in CML](https://community.cloudera.com/t5/Community-Articles/Using-Cloudera-Machine-Learning-for-Datalake-and-Iceberg/ta-p/336133)
