# Model Card
This work is part of the evaluation of the EEC1509 Machine Learning course of the Department of Computer Engineering and Automation Graduate Program in Electrical and Computer Engineering, Federal University of Rio Grande do Norte.

 - Developers: Francisval Guedes ([www.linkedin.com/in/francisval](https://www.linkedin.com/in/francisval-guedes-soares-6094772a)), Hareton Gomes ([www.linkedin.com/in/hareton](https://www.linkedin.com/in/hareton-ribeiro-gomes-11123a238/)).
 - Supervisor: Prof. Ivanovitch (https://www.linkedin.com/in/ivanovitchm/)


The model consists of a Multilayer Perceptron (MLP) for predicting bank marketing output results. It is an improvement of a previous work using [decision tree](https://github.com/francisvalguedes/bank_marketing/tree/master/classification).

## The data

The dataset is related with direct marketing campaigns of a Portuguese banking institution.
The *data* is from *May 2008 to November 2010* and contains information on an individual's ``marital, age, education, type of work, and more``.
The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be *('yes') or not ('no') subscribed*.

You can download the data from the [University of California, Irvine's website](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

Data attribute information:
   ##### Bank client data:   
   1. age (numeric)
   2. job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services") 
   3. marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
   4. education (categorical: "unknown","secondary","primary","tertiary")
   5. default: has credit in default? (binary: "yes","no")
   6. balance: average yearly balance, in euros (numeric) 
   7. housing: has housing loan? (binary: "yes","no")
   8. loan: has personal loan? (binary: "yes","no")
   ##### Related with the last contact of the current campaign:
   9. contact: contact communication type (categorical: "unknown","telephone","cellular") 
  10. day: last contact day of the month (numeric)
  11. month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
  12. duration: last contact duration, in seconds (numeric)
   ##### Other attributes:
  13. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
  14. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
  15. previous: number of contacts performed before this campaign and for this client (numeric)
  16. poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
  ##### Output variable (desired target):
  17. y - has the client subscribed a term deposit? (binary: "yes","no")

## Intended Use
This model is used in predicting the outcome of a banking marketing campaign, with an entire data pipeline incorporating Machine Learning fundamentals. The data pipeline is composed of the following stages:
 - File "etl.ipynb" - Extract Load Transform (ETL) - Fetch Data, EDA and Preprocessing
 - File "data_check_segregation.ipynb" - Data Check and Data Segregation 
 - File "train_test.ipynb" - Train and Test the model

In ETL item 1.2, an Exploratory Data Analysis (EDA) is carried out, to identify the main characteristics of the dataset with the objective of subsidizing appropriate treatment decisions. It was noted that the training data is imbalanced when considered the ``target`` variable and some features as( ``default``, ``loan``, ``contact``...)

## Model Details
A complete data pipeline was built using Google Colab, Visual Studio Code, Scikit-Learn, Keras, Tensorflow and Weights & Bias to train a MLP model. The big-picture of the data pipeline is shown below:

<img src="../figures/big_picture_model.png" width="600">


The accentuated imbalance of the output variable led the model to a poor performance in the prediction of the minority class, getting low *recall* and *F1* score metrics. Thus, the oversample technique was implemented in the training notebook, item 1.1 (Data preparation), using *SMOTENC* to data balance.

For prediction, 3 and 4 layer neural network models were analyzed. However, it was found that 3-layer models are sufficient for a good network performance.

For the sake of understanding, a hyperparameter-tuning was conducted using a Random Sweep of Wandb, and the best hyperparameters values adopted in the train were:

- layer_1 = 21,
- layer_2 = 43,
- learn_rate = 0.073,
- beta_1=0.99,
- beta_2=0.999,
- batch_size = 4096,
- epoch = 500,
- dropout1 = 1.0,
- dropout2 = 1.0,
- l2_1 = 0.00001,
- l2_2 = 0.0001,
- gradient_cliping = 0.5,
- bath_norm = 1,

With recovery of the best model by the F1 validation score metric at epoch 82.

## Evaluation Data
The dataset under study is split into Train and Test during the ``Segregate`` stage of the data pipeline. 70% of the clean data is used to Train and the remaining 30% to Test. Additionally, 30% of the Train data is used for validation purposes (hyperparameter-tuning). 

## Metrics
In order to follow the performance of machine learning experiments, the project marked certains stage outputs of the data pipeline as metrics. The metrics adopted are: [accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html), [f1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score), [precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score), [recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score).

To calculate the evaluations metrics is only necessary to run:

The follow results will be shown:

 **Stage [Run]**                        | **Accuracy** | **F1** | **Precision** | **Recall** | 
---------------------------------|--------------|--------|---------------|------------|
 Train [divine-oath-174](https://wandb.ai/mlops_ivan/decision_tree_bank/runs/43pj5775/overview?workspace=user-francisvalfgs) | 0.8961       | 0.3524 | 0.6486        | 0.2419     |  
 Test [comic-dragon-175](https://wandb.ai/mlops_ivan/decision_tree_bank/runs/mbpuwfbg/overview?workspace=user-francisvalfgs)  | 0.8961       | 0.345 | 0.6578        | 0.2338     |

## Caveats and Recommendations

