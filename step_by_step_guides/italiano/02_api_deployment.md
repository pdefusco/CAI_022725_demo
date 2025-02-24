## 02 Distribuzione API

#### Obiettivo

Questo documento spiega gli aspetti più importanti di 02_api_deployment.py.

#### Istruzioni per l'esecuzione del codice

Apri 02_api_deployment.py nella tua sessione CML e aggiorna le variabili DBNAME, STORAGE e CONNECTION_NAME alle righe 160-162 come indicato dal tuo HOL Lead.

Successivamente, premi il pulsante di riproduzione per eseguire l'intero script. Potrai osservare l'output del codice sul lato destro dello schermo.

#### Punti salienti del codice

* Riga 46: la classe "ModelDeployment" è importata dall'utilitario "mlops". Questo utilitario è stato collocato nella cartella "/home/cdsw".

* Riga 49: il client API CML è istanziato. L'API ti fornisce oltre 100 metodi Python per eseguire azioni come la creazione di progetti, l'avvio di job e molto altro. In questo esempio, l'API viene utilizzata per "list_projects()".

* Riga 62: il Client API viene passato come argomento all'istanza di ModelDeployment. L'utilitario mlops.py include alcuni metodi che estendono e sovrascrivono i metodi API. In genere, gli ingegneri di machine learning CML creano interfacce Python per costruire pipeline MLOps personalizzate in base ai loro casi d'uso.

* Riga 68: l'ultima Esplorazione Esperimento MLFlow viene utilizzata per registrare il Modello nel Registro MLFlow di CML.

* Righe 74, 78, 81: il Modello registrato viene utilizzato per creare una nuova Distribuzione Modello CML. Il Modello viene prima creato, poi costruito e infine distribuito.

#### Sommario

In questo laboratorio hai utilizzato CML APIv2 per eseguire azioni in modo programmatico all'interno degli spazi di lavoro CML. Puoi utilizzare l'API con semplici comandi curl CLI o con il Wrapper Python, che è una libreria preinstallata in ogni Runtime Cloudera CML. L'API può essere utilizzata sia da sistemi di terze parti esterni allo spazio di lavoro CML, sia all'interno dello spazio di lavoro CML.

Gli scienziati dei dati di CML sfruttano l'API per costruire pipeline MLOps. In questo laboratorio, hai utilizzato una semplice interfaccia Python per sovrascrivere i metodi API e costruire una pipeline MLOps standardizzata per registrare un'Esplorazione Esperimento nel Registro MLFlow e poi distribuire un Endpoint API per il servizio del modello.

#### Articoli Correlati

* Per saperne di più sui Distribuzioni di Modelli CML:
  * [Distribuzione Modello CML con MLFlow e APIv2 Articolo](https://community.cloudera.com/t5/Community-Articles/CML-Model-Deployment-with-MLFlow-and-APIv2/ta-p/385656)

* Per saperne di più sull'API CML:
  * [CML APIv2](https://docs.cloudera.com/machine-learning/cloud/api/topics/ml-api-v2.html)
  * [Riferimento API REST CML](https://docs.cloudera.com/machine-learning/cloud/rest-api-reference/index.html)
  * [CML API v2 AMP](https://github.com/cloudera/CML_AMP_APIv2)
