## 01 Treinamento XGBoost

#### Objetivo

Este documento explica os aspectos mais importantes do 01_train_xgboost.py.

#### Instruções para Execução do Código

Abra 01_train_xgboost.py na sua sessão CML e atualize as variáveis DBNAME, STORAGE e CONNECTION_NAME nas linhas 60-62 conforme orientado pelo seu líder HOL.

Em seguida, pressione o botão de reprodução para executar o script completo. Você poderá observar a saída do código no lado direito da tela.

Navegue até a aba Experimentos MLFlow e valide a criação do experimento. Abra a Execução do Experimento e familiarize-se com os metadados da execução do experimento. Na parte inferior da página, abra a pasta Artifacts e observe que as dependências do modelo foram rastreadas.

Em seguida, retorne à sessão CML e modifique o código:

* Na linha 74, substitua a seguinte linha de código:

```
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
```

por:

```
model = XGBClassifier(use_label_encoder=False, max_depth=4, eval_metric="logloss")
```

* Na linha 95, adicione:

```
mlflow.log_param("max_depth", 4)
```

Execute o script. Depois, retorne à página de Experimentos MLFlow e valide a nova Execução do Experimento. Compare os metadados da execução do experimento com a execução anterior e observe as mudanças.

#### Destaques do Código

* Linha 41: o pacote MLFlow é importado. O MLFlow está instalado por padrão nos projetos CML. Um plugin interno traduz os métodos MLFlow em métodos da API CML. Não é necessário instalar ou configurar o MLFlow para utilizar suas capacidades de rastreamento.

* Linhas 72 - 100: o código de treinamento do XGBoost é definido no contexto de uma Execução de Experimento MLFlow. O código XGBoost permanece inalterado. O método "mlflow.log_param()" é utilizado para registrar as métricas do modelo. O método "mlflow.log_model()" é utilizado para rastrear os artefatos do modelo na pasta "artifacts".

* Linha 120: o Cliente MLFlow é utilizado para interagir com os metadados da execução do experimento. Você pode usar o Cliente para pesquisar entre as execuções, comparar resultados e muito mais.

#### Resumo

Neste laboratório, você utilizou o MLFlow para rastrear execuções de experimentos na interface de experimentos, acessar dados das execuções dos experimentos de forma programática através do cliente MLFlow e registrar execuções no Registro MLFlow. Quando uma Execução de Experimento é rastreada, o MLFlow armazena automaticamente os artefatos e as dependências do modelo no backend.

O Registro é um componente separado do Espaço de Trabalho e atua como um ambiente de staging para mover opcionalmente modelos e dependências associadas de um Espaço de Trabalho para outro, por exemplo, em um padrão DEV para QA para PRD.

O MLFlow no CML não requer nenhuma instalação ou configuração por parte dos administradores ou usuários do CML. Está pré-instalado por padrão em todos os Espaços de Trabalho CML. O CML inclui um plugin especial que traduz chamadas da API MLFlow em rotinas da API CML v2. Você aprenderá mais sobre a API CML v2 na próxima seção.

#### Artigos Relacionados

* Para saber mais sobre MLFlow:
  * [Documentação MLFlow](https://mlflow.org/docs/latest/index.html)
  * [Registro de um modelo na interface MLFlow do CML](https://docs.cloudera.com/machine-learning/1.5.4/models/topics/ml-registering-model-using-ui.html)
  * [Documentação MLFlow CML](https://docs.cloudera.com/machine-learning/cloud/experiments/topics/ml-experiments-v2.html)
  * [Otimização de Hiperparâmetros com Experimentos no CML](https://community.cloudera.com/t5/Community-Articles/Tuning-Hyperparameters-with-Experiments-feature-on-Cloudera/ta-p/368654)

* Para saber mais sobre XGBoost:
  * [Documentação XGBoost](https://xgboost.readthedocs.io/en/stable/)
  * [XGBoost distribuído com PySpark no CML](https://community.cloudera.com/t5/Community-Articles/Distributed-XGBoost-with-PySpark-in-Cloudera-Machine/ta-p/375810)
  * [XGBoost CML com Dask AMP](https://github.com/cloudera/CML_AMP_Dask_on_CML)
