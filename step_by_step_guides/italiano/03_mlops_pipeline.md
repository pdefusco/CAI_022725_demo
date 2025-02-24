## 03 Pipeline MLOps

#### Obiettivo

Questo documento spiega gli aspetti più importanti di 03_newbatch.py, 04_train_xgboost.py e 05_api_redeployment.py.

#### Istruzioni per l'esecuzione del codice

Apri 03_newbatch.py, 04_train_xgboost.py e 05_api_redeployment.py nella tua sessione CML. Familiarizzati con il codice e aggiorna le variabili DBNAME, STORAGE e CONNECTION_NAME come indicato dal tuo responsabile HOL.

Non eseguire i script individualmente. Crea invece un lavoro CML per ciascuno di essi. Non eseguire ancora i lavori.

Crea il lavoro "New Batch" con le seguenti configurazioni:

```
Nome: New Batch Paul
Script: 03_newbatch.py
Editor: Workbench
Kernel: Python 3.9
Spark Add On: Spark 3.2 o 3.3
Edizione: Standard
Versione: 2024.02
Pianificazione: Manuale
Profilo Risorse: 2 vCPU / 4 GiB / 0 GPU
```

Crea il lavoro "Retrain XGBoost" con le seguenti configurazioni:

```
Nome: Retrain XGBoost Paul
Script: 04_train_xgboost.py
Editor: Workbench
Kernel: Python 3.9
Spark Add On: Spark 3.2 o 3.3
Edizione: Standard
Versione: 2024.02
Pianificazione: Dipendente da New Batch Paul
Profilo Risorse: 2 vCPU / 4 GiB / 0 GPU
```

Crea il lavoro "API Redeployment" con le seguenti configurazioni:

```
Nome: API Redeployment Paul
Script: 05_api_redeployment.py
Editor: Workbench
Kernel: Python 3.9
Spark Add On: Spark 3.2 o 3.3
Edizione: Standard
Versione: 2024.02
Pianificazione: Dipendente da Retrain XGBoost Paul
Profilo Risorse: 2 vCPU / 4 GiB / 0 GPU
```

Una volta creati tutti e tre i lavori, avvia manualmente il lavoro New Batch. Monitora l'esecuzione nella scheda Cronologia Lavori e osserva che, una volta completato, il lavoro successivo nel pipeline MLOps, Retrain XGBoost, viene avviato e infine il lavoro finale, API Redeployment, viene eseguito.

#### Punti salienti del codice

* 03_newbatch.py è quasi identico a 00_datagen.py.

* 04_train_xgboost.py è quasi identico a "01_train_xgboost.py". Tuttavia, alle righe 67-69, i metadati dello snapshot Iceberg sono memorizzati come variabili. Questi metadati vengono utilizzati alle righe 71-75 per eseguire una lettura incrementale, cioè caricare solo i dati dalla tabella Iceberg all'interno di un intervallo di tempo specificato. I metadati vengono quindi salvati come tag MLFlow durante l'esecuzione dell'esperimento.

* 05_api_redeployment.py include sia i metodi dell'utilitario mlops sia il codice per eseguire il pipeline MLOps. Questo è anche quasi identico al codice di "02_api_deployment.py".

#### Riepilogo

In questo laboratorio, hai utilizzato i lavori CML in combinazione con l'API CMLv2, Apache Iceberg e MLFlow per orchestrare un pipeline MLOps più avanzato. Con solo tre script e poche righe di codice, hai implementato un processo CI/CD standardizzato che aderisca alle migliori pratiche MLOps, inclusa la riproducibilità dei dati e dei modelli, l'auditabilità e l'esplicitabilità. Hai fatto ciò utilizzando componenti integrati e senza installazioni personalizzate.

* Nel primo lavoro, un nuovo batch di dati viene aggiunto alla tabella Iceberg Credit Card Transaction.

* Nel secondo lavoro, hai utilizzato Iceberg Time Travel per leggere i dati all'interno di limiti di tempo specificati - in altre parole, per accedere solo all'ultimo batch di dati - e poi riaddestrare lo stesso classificatore XGBoost con questi dati.

* Infine, nell'ultimo lavoro, hai ridistribuito una nuova versione dell'API Endpoint con l'ultima versione del modello.

#### Articoli Correlati

* Per saperne di più sui lavori CML:
  * [Creare un lavoro CML](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-job-c.html)
