## 03 Pipeline MLOps

#### Objetivo

Este documento explica os aspectos mais importantes de 03_newbatch.py, 04_train_xgboost.py e 05_api_redeployment.py.

#### Instruções para Execução do Código

Abra 03_newbatch.py, 04_train_xgboost.py e 05_api_redeployment.py na sua sessão CML. Familiarize-se com o código e atualize as variáveis DBNAME, STORAGE e CONNECTION_NAME conforme indicado pelo seu líder HOL.

Não execute os scripts individualmente. Crie um trabalho CML para cada um deles. Não execute os trabalhos ainda.

Crie o trabalho "New Batch" com as seguintes configurações:

```
Nome: New Batch Paul
Script: 03_newbatch.py
Editor: Workbench
Kernel: Python 3.9
Spark Add On: Spark 3.2 ou 3.3
Edição: Standard
Versão: 2024.02
Agendamento: Manual
Perfil de Recursos: 2 vCPU / 4 GiB / 0 GPU
```

Crie o trabalho "Retrain XGBoost" com as seguintes configurações:

```
Nome: Retrain XGBoost Paul
Script: 04_train_xgboost.py
Editor: Workbench
Kernel: Python 3.9
Spark Add On: Spark 3.2 ou 3.3
Edição: Standard
Versão: 2024.02
Agendamento: Dependente de New Batch Paul
Perfil de Recursos: 2 vCPU / 4 GiB / 0 GPU
```

Crie o trabalho "API Redeployment" com as seguintes configurações:

```
Nome: API Redeployment Paul
Script: 05_api_redeployment.py
Editor: Workbench
Kernel: Python 3.9
Spark Add On: Spark 3.2 ou 3.3
Edição: Standard
Versão: 2024.02
Agendamento: Dependente de Retrain XGBoost Paul
Perfil de Recursos: 2 vCPU / 4 GiB / 0 GPU
```

Depois de criar todos os três trabalhos, acione manualmente o trabalho New Batch. Monitore a execução na guia Histórico de Trabalhos e observe que, uma vez concluído, o próximo trabalho na pipeline MLOps, Retrain XGBoost, é acionado, e finalmente o último trabalho, API Redeployment, é executado.

#### Destaques do Código

* 03_newbatch.py é praticamente idêntico a 00_datagen.py.

* 04_train_xgboost.py é quase idêntico a "01_train_xgboost.py". No entanto, nas linhas 67-69, os metadados do snapshot Iceberg são armazenados como variáveis. Esses metadados são usados nas linhas 71-75 para realizar uma leitura incremental, ou seja, carregar apenas os dados da tabela Iceberg dentro de um intervalo de tempo especificado. Os metadados são então salvos como tags MLFlow durante a execução do experimento.

* 05_api_redeployment.py inclui tanto os métodos do utilitário mlops quanto o código para executar a pipeline MLOps. Isso também é quase idêntico ao código em "02_api_deployment.py".

#### Resumo

Neste laboratório, você usou os Trabalhos CML em conjunto com a API CMLv2, Apache Iceberg e MLFlow para orquestrar uma pipeline MLOps mais avançada. Com apenas três scripts e algumas linhas de código, você implementou um processo CI/CD padronizado que adere às melhores práticas MLOps, incluindo reprodutibilidade de dados e modelos, auditabilidade e explicabilidade. Você fez isso utilizando componentes integrados e sem instalações personalizadas.

* No primeiro trabalho, um novo lote de dados é adicionado à tabela Iceberg Credit Card Transaction.

* No segundo trabalho, você usou o Iceberg Time Travel para ler dados dentro de limites de tempo especificados - em outras palavras, para acessar apenas o lote mais recente de dados - e então re-treinou o mesmo classificador XGBoost com esses dados.

* Finalmente, no último trabalho, você redistribuiu uma nova versão do Endpoint da API com a versão mais recente do modelo.

#### Artigos Relacionados

* Para saber mais sobre Trabalhos CML:
  * [Criar um Trabalho CML](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-job-c.html)
