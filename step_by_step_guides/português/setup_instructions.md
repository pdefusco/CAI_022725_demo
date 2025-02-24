## Configuração

### Objetivo

Este documento fornece instruções para configurar o HOL em seu espaço de trabalho CML.

### Requisitos

* Espaço de trabalho CML na versão xxx com tipos de instância xxx
* Registro MLFlow CML
* Usuário CDP com direitos de administrador CML e configuração completa em Mappings IDBroker e Políticas Hadoop SQL / RAZ do Ranger.
* Perfil de recursos runtime CML de 2 vCPU e 4 ou 8 GiB

### Instruções de Configuração

1. Desdobre o Projeto CML do Repositório Git
2. Crie uma Sessão CML e Instale os Requisitos

#### 1. Desdobrar o Projeto CML do Repositório Git

Dentro do espaço de trabalho CML, crie um novo projeto e insira os seguintes parâmetros no formulário:

```
Nome do Projeto: MLOps HOL <username>
Visibilidade do Projeto: Privado ou Público
Configuração Inicial: Git -> https://github.com/pdefusco/CML_MLops_Banking_MLFlow.git
Runtimes :
  1. Remova todos os runtimes padrão.
  2. Selecione Opções Avançadas
  3. Selecione: Editor Workbench / Kernel Python 3.9 / Edição Standard / Versão 2024.02
```

![alt text](../../img/holbnk1.png)

![alt text](../../img/holbnk2.png)

#### 2. Crie uma Sessão CML e Instale os Requisitos

Inicie uma sessão CML com:

```
Editor: Workbench
Kernel: Python 3.9
Edição: Standard
Versão: 2024.02
Habilitar Spark: Versão 3.2 ou 3.3
Perfil de Recursos: 2 vCPU / 4 GiB Memória / 0 GPU
```

No prompt do lado direito, insira o seguinte comando:

```
!pip3 install -r requirements.txt
```

![alt text](../../img/holbnk3.png)

![alt text](../../img/holbnk4.png)

Uma vez que todos os pacotes tenham sido instalados, prossiga para as instruções em 00_datagen.

![alt text](../../img/holbnk5.png)
