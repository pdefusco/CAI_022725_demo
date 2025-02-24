## 04 Monitoraggio del Modello

#### Obiettivo

Questo documento spiega gli aspetti più importanti di 06_model_simulation.py e 07_model_monitoring.py.

#### Istruzioni per l'Esecuzione del Codice

Apri 06_model_simulation.py e 07_model_monitoring.py nella tua sessione CML. Familiarizzati con il codice e aggiorna le variabili DBNAME, STORAGE e CONNECTION_NAME come indicato dal tuo responsabile HOL.

Esegui 06_model_simulation.py e poi esci immediatamente dalla sessione per accedere al Deployment del Modello CML. Apri la scheda Monitoring e guarda il cruscotto di monitoraggio aggiornarsi in tempo reale man mano che le richieste vengono ricevute dal punto di fine del modello.

Successivamente, esegui 07_model_monitoring.py. Esplora i diagrammi di monitoraggio del modello sul lato destro della pagina. Come sta performando il tuo modello?

#### Punti Salienti del Codice

06_model_simulation.py crea più dati sintetici e utilizza il SDK CDSW per interagire con i punti di fine distribuiti. I dati vengono inviati al modello insieme alla verità di base per simulare un'ondata di richieste verso il punto di fine.

* Riga 41: il SDK cdsw è importato. Non è necessario installarlo.
* Righe 91-129: un metodo per inviare batch di richieste al punto di fine del Deployment del Modello specificato.
* Righe 134-154: un metodo per inviare la verità di base al punto di fine del Deployment del Modello specificato.

07_model_monitoring.py ti mostra come puoi monitorare le prestazioni del modello programmaticamente. CML dispone di un database Postgres chiamato "Model Metrics Store" che memorizza automaticamente i metadati per ogni richiesta da parte del modello distribuito. In questo script, il SDK cdsw è utilizzato di nuovo per leggere i metadati del modello e accedere al Model Metrics Store.

* Righe 66-67: l'ApiUtility viene istanziato per ottenere i metadati del modello. Il codice sorgente dell'utilitario si trova nella cartella src. Similmente all'utilitario "mlops" che hai utilizzato in 02_api_deployment.py, questo utilitario ti consente di costruire metodi personalizzati per ottenere i metadati del modello come necessario per il tuo caso d'uso.

* Riga 76: cdsw.read_metrics() viene utilizzato per leggere le richieste di modello tracciate dal Model Metrics Store.

* Seaborn e Pandas vengono utilizzati nel resto dello script per creare i grafici delle prestazioni del modello.

#### Riassunto

In questo laboratorio hai utilizzato il SDK cdsw per accedere alle richieste di previsione e alla verità di base memorizzata nel Model Metrics Store di CML per monitorare le prestazioni del modello (accuratezza).

Puoi sfruttare gli stessi strumenti in combinazione con i Jobs CML pianificati per creare sistemi di monitoraggio continuo e, similmente ai laboratori precedenti, distribuire programmaticamente nuove versioni del modello quando i criteri di prestazione non sono soddisfatti.

#### Articoli Correlati

* Per saperne di più sul monitoraggio delle metriche dei modelli in CML:
  * [Documentazione sulle Metriche dei Modelli](https://docs.cloudera.com/machine-learning/cloud/model-metrics/topics/ml-enabling-model-metrics.html)
  * [Governance dei Modelli](https://docs.cloudera.com/machine-learning/cloud/model-governance/topics/ml-enabling-model-governance.html)

* Per saperne di più sui Deployment e Monitoraggio dei Modelli CML:
  * [Creazione e Distribuzione di un Modello](https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-creating-and-deploying-a-model.html)
  * [Monitoraggio Continuo dei Modelli CML AMP](https://github.com/cloudera/CML_AMP_Continuous_Model_Monitoring)
