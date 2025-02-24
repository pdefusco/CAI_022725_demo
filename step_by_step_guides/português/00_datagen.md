## 00 Datagen

#### Objetivo

Este documento explica os aspectos mais importantes de 00_datagen.py.

#### Instruções para Execução do Código

Abra 00_datagen.py na sua Sessão CML e atualize a variáve CONNECTION_NAME na linha 162 conforme orientado pelo seu líder HOL.

Em seguida, pressione o botão de reprodução para executar o script inteiro. A saída do código está disponível no lado direito da tela.

#### Destaques do Código

* Linha 50: a biblioteca cml.data_v1 é importada. Esta biblioteca permite que você aproveite as Conexões de Dados CML para iniciar uma Sessão Spark e se conectar ao Data Lake. A DataConnection é usada nas linhas 103 - 109 no módulo "createSparkConnection".

* Linhas 64 - 95: o módulo "dataGen" é usado para criar dados sintéticos para o caso de uso de classificação. Observe os atributos dos dados que estão sendo criados, bem como seus tipos e intervalos de valores respectivos.

* Linhas 141 e 143: a API PySpark para Apache Iceberg é usada para criar ou adicionar dados a uma tabela no formato Iceberg a partir de um DataFrame PySpark.

#### Resumo

Neste laboratório, você usou as Conexões de Dados CML para pré-configurar o código básico para acesso a dados do CDP e fontes de dados de terceiros, incluindo Postgres, SQLServer e até mesmo Snowflake. Você pode personalizar as Conexões de Dados para padronizar e simplificar as configurações de Acesso aos Dados, além de restringir o acesso a sistemas de terceiros.

A API PySpark Iceberg permite criar tabelas Iceberg a partir de DataFrames Spark. É semelhante e tão simples quanto a API PySpark padrão para interagir com tabelas Hive.

### Artigos Relacionados

* Para saber mais sobre Conexões de Dados CML:
  * [Artigo sobre Conexões de Dados](https://community.cloudera.com/t5/Community-Articles/New-Feature-in-Cloudera-Machine-Learning-Data-Connections/ta-p/336775)
  * [Conexões de Dados Personalizadas](https://docs.cloudera.com/machine-learning/cloud/mlde/topics/ml-custom-data-conn-create.html)
  * [Artigo sobre Conexões de Dados Personalizadas](https://community.cloudera.com/t5/Community-Articles/Using-Custom-Data-Connections-in-Cloudera-Machine-Learning/ta-p/379132)

* Para saber mais sobre Apache Iceberg:
  * [Documentação do Apache Iceberg](https://iceberg.apache.org/docs/1.5.2/)
  * [Apache Iceberg com Spark](https://iceberg.apache.org/docs/1.5.2/spark-getting-started/)
  * [Apache Iceberg no Cloudera](https://www.cloudera.com/open-source/apache-iceberg.html)
  * [Apache Iceberg no CML](https://community.cloudera.com/t5/Community-Articles/Using-Cloudera-Machine-Learning-for-Datalake-and-Iceberg/ta-p/336133)
