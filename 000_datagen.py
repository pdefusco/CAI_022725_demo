#****************************************************************************
# (C) Cloudera, Inc. 2020-2023
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

import os
import numpy as np
import pandas as pd
from datetime import datetime
import dbldatagen as dg
import dbldatagen.distributions as dist
from dbldatagen import FakerTextFactory, DataGenerator, fakerText
from pyspark.sql import SparkSession
from faker.providers import bank, credit_card, currency
from pyspark.sql.types import LongType, FloatType, IntegerType, StringType, \
                              DoubleType, BooleanType, ShortType, \
                              TimestampType, DateType, DecimalType, \
                              ByteType, BinaryType, ArrayType, MapType, \
                              StructType, StructField
import cml.data_v1 as cmldata

class BankDataGen:

    '''Class to Generate Banking Data'''

    def __init__(self, username, dbname, connectionName):
        self.username = username
        self.dbname = dbname
        self.connectionName = connectionName

    def transactionsDataGen(self, shuffle_partitions_requested = 5, partitions_requested = 5, data_rows = 2000):
        """
        Method to create a Spark DF of fictitious credit card transactions
        """

        # setup use of Faker
        FakerTextUS = FakerTextFactory(locale=['en_US'], providers=[bank])

        # partition parameters etc.
        self.spark.conf.set("spark.sql.shuffle.partitions", shuffle_partitions_requested)

        fakerDataspec = (DataGenerator(self.spark, rows=data_rows, partitions=partitions_requested)
                    .withColumn("credit_card_number", "long", minValue=3674567891195999, maxValue=3674567891198000, uniqueValues=2000, step=1)
                    .withColumn("credit_card_provider", text=FakerTextUS("credit_card_provider") )
                    .withColumn("transaction_type", "string", values=["purchase", "cash_advance"], random=True, weights=[9, 1])
                    .withColumn("event_ts", "timestamp", begin="2023-01-01 01:00:00",end="2023-12-31 23:59:00",interval="1 minute", random=True)
                    .withColumn("longitude", "float", minValue=-125, maxValue=-66.9345, random=True)
                    .withColumn("latitude", "float", minValue=24.3963, maxValue=49.3843, random=True)
                    .withColumn("transaction_currency", values=["USD", "EUR", "KWD", "BHD", "GBP", "CHF", "MEX"])
                    .withColumn("transaction_amount", "decimal", minValue=0.01, maxValue=30000, random=True)
                    .withColumn("transaction", StructType([StructField('transaction_currency', StringType()), StructField('transaction_amount', StringType()), StructField('transaction_type', StringType())]),
                    expr="named_struct('transaction_currency', transaction_currency, 'transaction_amount', transaction_amount, 'transaction_type', transaction_type)",
                    baseColumn=['transaction_currency', 'transaction_currency', 'transaction_type'])
                    .withColumn("transaction_geolocation", StructType([StructField('latitude',StringType()), StructField('longitude', StringType())]),
                      expr="named_struct('latitude', latitude, 'longitude', longitude)",
                      baseColumn=['latitude', 'longitude'])
                    )

        df = fakerDataspec.build()

        df = df.drop(*('latitude', 'longitude', 'transaction_currency', 'transaction_amount', 'transaction_type'))

        df = df.withColumn("credit_card_number", df["credit_card_number"].cast("string"))

        return df

    def piiDataGen(self, shuffle_partitions_requested = 5, partitions_requested = 5, data_rows = 2000):
        """
        Method to create a Spark DF of fictitious PII
        """

        # setup use of Faker
        FakerTextUS = FakerTextFactory(locale=['en_US'], providers=[bank])

        # partition parameters etc.
        self.spark.conf.set("spark.sql.shuffle.partitions", shuffle_partitions_requested)

        fakerDataspec = (DataGenerator(self.spark, rows=data_rows, partitions=partitions_requested)
                    .withColumn("name", text=FakerTextUS("name") )
                    .withColumn("address", text=FakerTextUS("address" ))
                    .withColumn("address_longitude", "float", minValue=-125, maxValue=-66.9345, random=True)
                    .withColumn("address_latitude", "float", minValue=24.3963, maxValue=49.3843, random=True)
                    .withColumn("email", text=FakerTextUS("ascii_company_email") )
                    .withColumn("aba_routing", text=FakerTextUS("aba" ))
                    .withColumn("bank_country", text=FakerTextUS("bank_country") )
                    .withColumn("account_no", text=FakerTextUS("bban" ))
                    .withColumn("int_account_no", text=FakerTextUS("iban") )
                    .withColumn("swift11", text=FakerTextUS("swift11" ))
                    .withColumn("credit_card_number", "long", minValue=3674567891195999, maxValue=3674567891197999, uniqueValues=1000, random=True, randomSeed=4)
                    )

        df = fakerDataspec.build()
        df = df.withColumn("credit_card_number", df["credit_card_number"].cast("string"))

        return df

    def createSparkConnection(self):
        """
        Method to create a Spark Connection using CML Data Connections
        """

        from pyspark import SparkContext
        SparkContext.setSystemProperty('spark.executor.cores', '2')
        SparkContext.setSystemProperty('spark.executor.memory', '4g')

        import cml.data_v1 as cmldata
        conn = cmldata.get_connection(self.connectionName)
        spark = conn.get_spark_session()

        return spark


    def saveFileToCloud(self, df):
        """
        Method to save credit card transactions df as csv in cloud storage
        """
        #df.write.format("csv").mode('overwrite').save(self.storage + "/bank_fraud_demo/" + self.username)
        pass


    def createDatabase(self, spark):
        """
        Method to create database before data generated is saved to new database and table
        """

        spark.sql("CREATE DATABASE IF NOT EXISTS {}".format(self.dbname))

        print("SHOW DATABASES LIKE '{}'".format(self.dbname))
        spark.sql("SHOW DATABASES LIKE '{}'".format(self.dbname)).show()


    def createOrReplace(self, df):
        """
        Method to create or append data to the BANKING TRANSACTIONS table
        The table is used to simulate batches of new data
        The table is meant to be updated periodically as part of a CML Job
        """

        try:
            df.writeTo("{0}.CC_TRX_{1}".format(self.dbname, self.username))\
              .using("iceberg").tableProperty("write.format.default", "parquet").append()

        except:
            df.writeTo("{0}.CC_TRX_{1}".format(self.dbname, self.username))\
                .using("iceberg").tableProperty("write.format.default", "parquet").createOrReplace()


    def validateTable(self, spark):
        """
        Method to validate creation of table
        """
        print("SHOW TABLES FROM '{}'".format(self.dbname))
        spark.sql("SHOW TABLES FROM {}".format(self.dbname)).show()


def main():

    USERNAME = os.environ["PROJECT_OWNER"]
    DBNAME = "BNK_MLOPS_HOL_"+USERNAME
    CONNECTION_NAME = "paul-november-aw-dl"

    # Instantiate BankDataGen class
    dg = BankDataGen(USERNAME, DBNAME, CONNECTION_NAME)

    # Create CML Spark Connection
    spark = dg.createSparkConnection()

    # Create Banking Transactions DF
    df = dg.dataGen(spark)

    # Create Spark Database
    dg.createDatabase(spark)

    # Create Iceberg Table in Database
    dg.createOrReplace(df)

    # Validate Iceberg Table in Database
    dg.validateTable(spark)


if __name__ == '__main__':
    main()
