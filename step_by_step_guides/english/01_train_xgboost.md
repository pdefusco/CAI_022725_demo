## 01 Train XGBoost

#### Objective

This document explains the most important aspects of 01_train_xgboost.py.

#### Instructions for Code Execution

Open 01_train_xgboost.py in your CML Session and update the DBNAME, STORAGE, and CONNECTION_NAME variables at lines 60-62 as instructed by your HOL Lead.

Next, press the play button in order to run the whole script. You will be able to observe code output on the right side of your screen.

Navigate to the MLFLow Experiments tab and validate experiment creation. Open the Experiment Run and familiarize yourself with Experiment Run metadata. At the bottom of the page, open the Artifacts folder and notice that model dependencies have been tracked.

Next, return to the CML Session and modify the code:

* At line 74, replace the following line of code:

```
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
```

with:

```
model = XGBClassifier(use_label_encoder=False, max_depth=4, eval_metric="logloss")
```

 * At line 95, add:

 ```
mlflow.log_param("max_depth", 4)
 ```

Run the script. Then return to the MLFLow Experiments page and validate the new Experiment Run. Compare Experiment Run metadata with the previous run and notice it has changed.

#### Code Highlights

* Line 41: the MLFlow package is imported. MLFlow is installed in CML Projects by default. An internal Plugin tranaslates MLFlow methods to CML API methods. There is no need to install or configure MLFlow in order to use its Tracking capabilities.

* Lines 72 - 100: XGBoost training code is defined within the context of an MLFlow Experiment Run. XGBoost code is otherwise unchanged. The "mlflow.log_param()" method is used to log model metrics. The "mlflow.log_model()" method is used to track model artifacts in the "artifacts" folder.

* Line 120: the MLFlow Client is used to interact with Experiment Run metadata. You can use the Client to search through runs, compare results, and much more.

#### Summary

In this lab used MLFlow to track experiment runs in the Experiments UI, access experiment rund ata programmatically via the MLFlow Client, and register Runs to the MLFlow Registry. When an Exoeriment Run is tracked, MLFlow automatically stores model artifacts and dependencies in the backend.  

The Registry is a separate component from the Workspace and acts as a staging environment for optionally moving models and associated dependencies from one Workspace to another, for example in a DEV to QA to PRD pattern.  

MLFlow in CML does not require any installation or configurations on the part of the CML Admins or Users. It is preinstalled by default in every CML Workspace. CML includes a special Plugin that translates MLFlow API calls to CML API v2 routines. You will learn more about CML API v2 in the next section.

#### Related Articles

* To learn more about MLFlow:
  * [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)
  * [Registering a Model in the CML MLFlow UI](https://docs.cloudera.com/machine-learning/1.5.4/models/topics/ml-registering-model-using-ui.html)
  * [CML MLFlow Documentation](https://docs.cloudera.com/machine-learning/cloud/experiments/topics/ml-experiments-v2.html)
  * [Tuning Hyperparameters with Experiments on CML](https://community.cloudera.com/t5/Community-Articles/Tuning-Hyperparameters-with-Experiments-feature-on-Cloudera/ta-p/368654)

* To learn more about XGBoost:
  * [XGboost Documentation](https://xgboost.readthedocs.io/en/stable/)
  * [Distributed XGBoost with PySpark in CML](https://community.cloudera.com/t5/Community-Articles/Distributed-XGBoost-with-PySpark-in-Cloudera-Machine/ta-p/375810)
  * [CML XGBoost with Dask AMP](https://github.com/cloudera/CML_AMP_Dask_on_CML)
