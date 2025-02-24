## Configuración

### Objetivo

Este documento proporciona instrucciones para configurar el HOL en tu espacio de trabajo de CML.

### Requisitos

* Espacio de trabajo de CML en versión xxx con tipos de instancia xxx
* Registro de MLFlow de CML
* Usuario de CDP con derechos de administrador de CML y configuración completa en Mappings de IDBroker y Políticas relacionadas con Ranger Hadoop SQL / RAZ.
* Perfil de Recursos del Runtime de CML de 2 vCPU y 4 o 8 GiB

### Instrucciones de Configuración

1. Desplegar el Proyecto de CML desde el Repositorio Git
2. Crear una Sesión de CML e Instalar Requisitos

#### 1. Desplegar el Proyecto de CML desde el Repositorio Git

Desde dentro del espacio de trabajo de CML, crea un nuevo proyecto e ingresa los siguientes parámetros en el formulario:

```
Nombre del Proyecto: MLOps HOL <nombre de usuario>
Visibilidad del Proyecto: Privado o Público
Configuración Inicial: Git -> https://github.com/pdefusco/CML_MLops_Banking_MLFlow.git
Runtimes:
  1. Elimina todos los runtimes predeterminados.
  2. Selecciona Opciones Avanzadas
  3. Selecciona: Editor de Workbench / Kernel Python 3.9 / Edición Standard / Versión 2024.02
```

![alt text](../../img/holbnk1.png)

![alt text](../../img/holbnk2.png)

#### 2. Crear una Sesión de CML e Instalar Requisitos

Lanza una sesión de CML con:

```
Editor: Workbench
Kernel: Python 3.9
Edición: Standard
Versión: 2024.02
Habilitar Spark: Versión 3.2 o 3.3
Perfil de Recursos: 2 vCPU / 4 GiB Mem / 0 GPU
```

En el prompt en el lado derecho, ingresa el siguiente comando:

```
!pip3 install -r requirements.txt
```

![alt text](../../img/holbnk3.png)

![alt text](../../img/holbnk4.png)

Una vez que todos los paquetes hayan sido instalados, procede con las instrucciones en 00_datagen.

![alt text](../../img/holbnk5.png)
