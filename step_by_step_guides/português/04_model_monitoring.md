## 04 Monitoramento do Modelo

#### Objetivo

Este documento explica os aspectos mais importantes de 06_model_simulation.py e 07_model_monitoring.py.

#### Instruções para Execução do Código

Abra 06_model_simulation.py e 07_model_monitoring.py em sua Sessão CML. Familiarize-se com o código e atualize as variáveis DBNAME, STORAGE e CONNECTION_NAME conforme instruído pelo seu líder HOL.

Execute 06_model_simulation.py e, em seguida, saia imediatamente da Sessão para acessar o Deployment do Modelo CML. Abra a guia Monitoring e observe o painel de monitoramento ser atualizado em tempo real à medida que as solicitações são recebidas pelo ponto de extremidade do modelo.

Em seguida, execute 07_model_monitoring.py. Explore os diagramas de monitoramento do modelo no lado direito da página. Como está o desempenho do seu modelo?

#### Principais Pontos do Código

06_model_simulation.py cria mais dados sintéticos e utiliza o SDK CDSW para interagir com os pontos de extremidade implantados. Os dados são enviados ao modelo junto com a verdade fundamental para simular uma onda de solicitações para o ponto de extremidade.

* Linha 41: o SDK cdsw é importado. Não é necessário instalá-lo.
* Linhas 91-129: um método para enviar lotes de solicitações para o ponto

 de extremidade do Deployment do Modelo especificado.
* Linhas 134-154: um método para enviar a verdade fundamental para o ponto de extremidade do Deployment do Modelo especificado.

07_model_monitoring.py mostra como você pode monitorar o desempenho do modelo programaticamente. O CML possui um banco de dados Postgres chamado "Model Metrics Store" que armazena automaticamente metadados para cada solicitação por modelo implantado. Neste script, o SDK cdsw é usado novamente para ler os metadados do modelo e acessar o Model Metrics Store.

* Linhas 66-67: o ApiUtility é instanciado para obter os metadados do modelo. O código-fonte do utilitário está localizado na pasta src. Da mesma forma que o utilitário "mlops" usado em 02_api_deployment.py, este utilitário permite que você construa métodos personalizados para obter os metadados do modelo conforme necessário para o seu caso de uso.

* Linha 76: cdsw.read_metrics() é usado para ler as solicitações de modelo rastreadas do Model Metrics Store.

* Seaborn e Pandas são usados ao longo do restante do script para criar os gráficos de desempenho do modelo.

#### Resumo

Neste laboratório, você usou o SDK cdsw para acessar solicitações de previsão e a verdade fundamental armazenada no Model Metrics Store do CML para monitorar o desempenho do modelo (precisão).

Você pode aproveitar as mesmas ferramentas em conjunto com Jobs CML agendados para criar sistemas de monitoramento contínuo e, de forma semelhante aos laboratórios anteriores, implantar programaticamente novas versões do modelo quando os critérios de desempenho não forem atendidos.

#### Artigos Relacionados

* Para saber mais sobre o rastreamento de métricas de modelos no CML:
  * [Documentação sobre Métricas de Modelos](https://docs.cloudera.com/machine-learning/cloud/model-metrics/topics/ml-enabling-model-metrics.html)
  * [Governança de Modelos](https://docs.cloudera.com/machine-learning/cloud/model-governance/topics/ml-enabling-model-governance.html)

* Para saber mais sobre o Deployment e Monitoramento de Modelos CML:
  * [Criando e Implantando um Modelo](https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-creating-and-deploying-a-model.html)
  * [Monitoramento Contínuo de Modelos CML AMP](https://github.com/cloudera/CML_AMP_Continuous_Model_Monitoring)
