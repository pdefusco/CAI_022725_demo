## 01 Addestramento XGBoost

#### Obiettivo

Questo documento spiega gli aspetti più importanti di 01_train_xgboost.py.

#### Istruzioni per l'esecuzione del codice

Apri 01_train_xgboost.py nella tua sessione CML e aggiorna le variabili DBNAME, STORAGE e CONNECTION_NAME alle righe 60-62 come indicato dal tuo responsabile HOL.

Successivamente, premi il pulsante di riproduzione per eseguire l'intero script. Sarai in grado di osservare l'output del codice sul lato destro dello schermo.

Vai alla scheda Esperimenti MLFLow e convalida la creazione dell'esperimento. Apri l'Esecuzione dell'Esperimento e familiarizzati con i metadati dell'esecuzione dell'esperimento. In fondo alla pagina, apri la cartella Artifacts e nota che le dipendenze del modello sono state tracciate.

Poi, torna alla sessione CML e modifica il codice:

* Alla riga 74, sostituisci la seguente riga di codice:

```
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
```

con:

```
model = XGBClassifier(use_label_encoder=False, max_depth=4, eval_metric="logloss")
```

* Alla riga 95, aggiungi:

```
mlflow.log_param("max_depth", 4)
```

Esegui lo script. Poi torna alla pagina degli Esperimenti MLFLow e convalida la nuova Esecuzione dell'Esperimento. Confronta i metadati dell'esecuzione dell'esperimento con quelli dell'esecuzione precedente e nota le modifiche.

#### Punti salienti del codice

* Riga 41: il pacchetto MLFlow è importato. MLFlow è installato di default nei progetti CML. Un plugin interno traduce i metodi MLFlow in metodi dell'API CML. Non è necessario installare o configurare MLFlow per utilizzare le sue capacità di tracciamento.

* Righe 72 - 100: il codice di addestramento XGBoost è definito nel contesto di un'Esecuzione di Esperimento MLFlow. Il codice XGBoost rimane invariato. Il metodo "mlflow.log_param()" è utilizzato per registrare le metriche del modello. Il metodo "mlflow.log_model()" è utilizzato per tracciare gli artefatti del modello nella cartella "artifacts".

* Riga 120: il Client MLFlow è utilizzato per interagire con i metadati dell'esecuzione dell'esperimento. Puoi utilizzare il Client per cercare tra le esecuzioni, confrontare i risultati e molto altro.

#### Riassunto

In questo laboratorio, hai utilizzato MLFlow per tracciare le esecuzioni degli esperimenti nell'interfaccia degli esperimenti, accedere ai dati delle esecuzioni degli esperimenti in modo programmatico tramite il client MLFlow e registrare le esecuzioni nel Registro MLFlow. Quando un'Esecuzione dell'Esperimento viene tracciata, MLFlow memorizza automaticamente gli artefatti e le dipendenze del modello nel backend.

Il Registro è un componente separato dallo spazio di lavoro e funge da ambiente di staging per spostare eventualmente modelli e dipendenze associate da uno spazio di lavoro a un altro, ad esempio in uno schema DEV a QA a PRD.

MLFlow in CML non richiede alcuna installazione o configurazione da parte degli amministratori o degli utenti CML. È preinstallato di default in ogni spazio di lavoro CML. CML include un plugin speciale che traduce le chiamate API MLFlow in routine dell'API CML v2. Imparerai di più sull'API CML v2 nella sezione successiva.

#### Articoli Correlati

* Per saperne di più su MLFlow:
  * [Documentazione MLFlow](https://mlflow.org/docs/latest/index.html)
  * [Registrazione di un modello nell'interfaccia utente MLFlow CML](https://docs.cloudera.com/machine-learning/1.5.4/models/topics/ml-registering-model-using-ui.html)
  * [Documentazione MLFlow CML](https://docs.cloudera.com/machine-learning/cloud/experiments/topics/ml-experiments-v2.html)
  * [Ottimizzazione degli iperparametri con le esperienze su CML](https://community.cloudera.com/t5/Community-Articles/Tuning-Hyperparameters-with-Experiments-feature-on-Cloudera/ta-p/368654)

* Per saperne di più su XGBoost:
  * [Documentazione XGboost](https://xgboost.readthedocs.io/en/stable/)
  * [XGBoost distribuito con PySpark in CML](https://community.cloudera.com/t5/Community-Articles/Distributed-XGBoost-with-PySpark-in-Cloudera-Machine/ta-p/375810)
  * [XGBoost CML con Dask AMP](https://github.com/cloudera/CML_AMP_Dask_on_CML)
