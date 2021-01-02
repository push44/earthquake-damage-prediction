# Predicting earthquake damage grade (Multiclass classification)
![earthquake](https://s3.amazonaws.com/drivendata-public-assets/nepal-quake-bm-2.JPG)

## Data source:
![Driven Data](https://www.drivendata.org/competitions/57/nepal-earthquake/)

## Problem statement:
According to Wikipwdia The April 2015 Nepal earthquake killed nearly 9,000 people and injured nearly 22,000. It occurred at 11:56 Nepal Standard Time on 25 April 2015, with a magnitude of 7.8Mw. But in this project we try to predict the building damage grade that are classified in three categories.
<ol>
  <li>Low damage</li>
  <li>Medium damage</li>
  <li>Complete destriction</li>
</ol>

## Files:
```bash
├── models
│   ├── binary_logistic_reg.pickle
│   └── decision_tree_clf.pickle
├── notebooks
│   ├── EDA.ipynb
│   └── EDA.pdf
├── README.md
└── src
    ├── analyze_and_visuallize.py
    ├── create_feature_interaction_target_encoding.py
    ├── create_folds.py
    ├── feature_selection.py
    ├── model_dispatcher.py
    ├── predict.py
    ├── __pycache__
    │   ├── model_dispatcher.cpython-38.pyc
    │   └── target_encoding.cpython-38.pyc
    ├── target_encoding.py
    └── train_cv.py
 ```

## Folder Description:
<ul>
  <li>models: This folder contains pickle files that contains trained models</li>
  <li>notebooks: This folder contains any jupyter notebooks such EDA.ipynb</li>
  <li>src: This fodler contains all .py scripts</li>
</ul>

## src Files Description:
<ul>
  <li>analyze_and_visuallize.py: This script analyze the feature selection output and save visuallization in .png format</li>
  <li>create_feature_interaction_target_encoding.py: As the name suggest this script creates feature interaction and performs mean target encoding on the feature interaction</li>
  <li>create_folds.py: This is the first script of the project where we create K (here 10) folds for cross-validation</li>
  <li>model_dispacther.py: This script provides all the required machine learning models for the project</li>
  <li>predict.py: This script trains machine learning model on the entire train data set and stores them in models folder. In the same script we load trained models from models folder to make final prediction on the trained dataset</li>
  <li>target_encoding.py: This script holds mean target encoding script</li>
  <li>train_cv.py: This script is used for training k fold model and getting evaluation score on a hold-out-dataset</li>
</ul>

## Data Description:
<ul>
  <li>Integer features: 31</li>
  <li>Categorical features: 8</li>
  <li>Binary features: 22</li>
</ul>

<b>Target distribution</b>:
![alt_text](/images/target_dist.png)

<br>

<b>Numeric features correlation plot:</b>
![alt_text](/images/correlation_map.png)

<br>

<b>Categorical feature description:</b>
![alt_text](/images/categorical_feat.png)

<br>

<b>Binary features:</b>
![alt_text](/images/cat_feat_target_var.png)


## Evaluation metrics:

<img src="https://render.githubusercontent.com/render/math?math=F_{micro} = \frac{2.P_{micro}.R_{micro}}{P_{micro}+R_{micro}}">

## Models:

| Model Description                                           | Train score  | CV score  | Test score  |
|-------------------------------------------------------------|--------------|-----------|-------------|
| Logistic reg + all categorical (OHE) + Binary features      | 0.5761       | 0.5761    | 0.5817      |
| Decision tree (numeric features) + logistic regression (categorical features)   | 0.6354       | 0.6348    | 0.6346      |
| Random forest (numeric features) + logistic regression (categorical features)   | 0.6656       | 0.6617    | 0.6577      |
| Random forest (numeric features) + logistic regression (categorical features) + Mean Target encoding   | 0.7022       | 0.7020    | 0.7029      |
| Random forest (numeric features) + logistic regression (categorical features) + Mean Target encoding + Feature interaction   | 0.7171       | 0.7137    | 0.7152      |

## Conclusion:
<ul>
  <li>From the evaluation metric we can say current model is 71.52% accurate on the test data. Hence, local government can make use of the important fetures to make regulatory decisions for the constructions of the building in selected area.</li>
  <li>In every selected feature interaction one of the three geo level id features exist. In particular geo level id 2 and geo level id 3 are important features. These two features gives granular information about the geological position of a building. Indicating some geological locations are safer than others.</li>
  <li>Some of the other important features are the foundation types and construction types. Indicating some types are safer to use than the others.</li>
  <li>Based on these informations, local government can take neccessary actions in the targetted area to prevent larger building damages in case future disasters.</li>
<ul>
