## 02 Implantação de API

#### Objetivo

Este documento explica os aspectos mais importantes do 02_api_deployment.py.

#### Instruções para Execução do Código

Abra o 02_api_deployment.py na sua sessão CML e familiarize-se com o código. Em seguida, abra o mlops.py e também familiarize-se com o código.

Depois, volte para o 02_api_deployment.py e pressione o botão de reprodução para executar o script completo. Você poderá observar a saída do código no lado direito da tela.

#### Destaques do Código

* Linha 46: a classe "ModelDeployment" é importada do utilitário "mlops". Este utilitário foi colocado na pasta "/home/cdsw".

* Linha 49: o cliente API CML é instanciado. A API fornece mais de 100 métodos Python para executar ações como criar projetos, iniciar jobs e muito mais. Neste exemplo, a API é usada para "list_projects()".

* Linha 62: o Client API é passado como argumento para a instância de ModelDeployment. O utilitário mlops.py inclui alguns métodos que estendem e substituem os métodos da API. Normalmente, os engenheiros de Machine Learning da CML criam Interfaces Python para construir pipelines MLOps personalizados de acordo com seu caso de uso.

* Linha 68: a última Execução de Experimento MLFlow é usada para registrar o Modelo no Registro MLFlow da CML.

* Linhas 74, 78, 81: o Modelo registrado é usado para criar uma nova Implantação de Modelo CML. O Modelo é primeiro criado, depois construído e, finalmente, implantado.

#### Resumo

Neste laboratório, você usou o CML APIv2 para executar ações de forma programática dentro dos Workspaces CML. Você pode usar a API com comandos curl CLI simples ou com o Wrapper Python, que é uma biblioteca pré-instalada em todos os Runtimes Cloudera CML. A API pode ser usada tanto de sistemas de terceiros externos ao Workspace CML quanto dentro do próprio Workspace CML.

Os Cientistas de Dados da CML aproveitam a API para construir pipelines MLOps. Neste laboratório, você usou uma Interface Python simples para substituir métodos da API e construir um pipeline MLOps padronizado para registrar uma Execução de Experimento no Registro MLFlow e, em seguida, implantar um Endpoint API para servir o modelo.

#### Artigos Relacionados

* Para saber mais sobre Implantações de Modelos CML:
  * [Implantação de Modelo CML com MLFlow e APIv2 Artigo](https://community.cloudera.com/t5/Community-Articles/CML-Model-Deployment-with-MLFlow-and-APIv2/ta-p/385656)

* Para saber mais sobre a API CML:
  * [CML APIv2](https://docs.cloudera.com/machine-learning/cloud/api/topics/ml-api-v2.html)
  * [Referência API REST CML](https://docs.cloudera.com/machine-learning/cloud/rest-api-reference/index.html)
  * [CML API v2 AMP](https://github.com/cloudera/CML_AMP_APIv2)
