# Predicting earthquake damage grade (Multiclass classification)
![earthquake](https://s3.amazonaws.com/drivendata-public-assets/nepal-quake-bm-2.JPG)

## A) Data source:
![Driven Data](https://www.drivendata.org/competitions/57/nepal-earthquake/)

## B) Problem statement:
According to Wikipwdia The April 2015 Nepal earthquake killed nearly 9,000 people and injured nearly 22,000. It occurred at 11:56 Nepal Standard Time on 25 April 2015, with a magnitude of 7.8Mw. But in this project we try to predict the building damage grade that are classified in three categories.
<ol>
  <li>Low damage</li>
  <li>Medium damage</li>
  <li>Complete destriction</li>
</ol>

## C) Files:
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

## D) Folder Description:
<ul>
  <li>models: This folder contains pickle files that contains trained models</li>
  <li>notebooks: This folder contains any jupyter notebooks such EDA.ipynb</li>
  <li>src: This fodler contains all .py scripts</li>
</ul>

## E) src Files Description:
<ul>
  <li>analyze_and_visuallize.py: This script analyze the feature selection output and save visuallization in .png format</li>
  <li>create_feature_interaction_target_encoding.py: As the name suggest this script creates feature interaction and performs mean target encoding on the feature interaction</li>
  <li>create_folds.py: This is the first script of the project where we create K (here 10) folds for cross-validation</li>
  <li>model_dispacther.py: This script provides all the required machine learning models for the project</li>
  <li>predict.py: This script trains machine learning model on the entire train data set and stores them in models folder. In the same script we load trained models from models folder to make final prediction on the trained dataset</li>
  <li>target_encoding.py: This script holds mean target encoding script</li>
  <li>train_cv.py: This script is used for training k fold model and getting evaluation score on a hold-out-dataset</li>
</ul>

## D) Data Description:
<ul>
  <li>Integer features: 8 (geo level 1 id, age, area percentage, height percentage, etc.)</li>
  <li>Categorical features: 8 (landsurfce condition, foundation type, roof type, etc.)</li>
  <li>Binary features: 22 (has superstructure adobe mud, has superstructure mud mortar stone, has secondary use agriculture, etc.)</li>
</ul>

<b>D.1 Target distribution</b>:
![alt_text](/images/target_dist.png)

The target distribution is uneven and we have to take care of this situation while cross-validation and model training. Cross-validation is the first step towards solving any machine learning problem where we assgin fold value to every sample in our training dataset.

<br>

<b>D.2) Numeric features correlation plot:</b>
![alt_text](/images/correlation_map.png)

The plot above shows the heatmap of the pearson correlation coefficients of all the numeric features. We can see some of the feature pairs are highly correlated such as (heigh_percentage, count_floors_pre_eq), (has_secondary_use_agriculture, has_secondary_use). Surely, we do not want to through entire feature set into the model and must do feature engineering steps before building a model.

<br>

<b>D.3) Categorical feature description:</b>
![alt_text](/images/categorical_feat.png)

There are only eight categorical feature in the feature set and the above plot explains the information we need to know about these eight features, as following:
<ul>
  <li>The first plot above indicates the level of categories in each feature (i.e., total possible number of values each category can take).</li>
  <li>The second plot above indicates the variance of the taget feature (damage grade prediction label) grouped by a feature. Note having low variance indicates target feature is more likely predictable than a feature having high variance.</li>
  <li>The third plot above shows the box-plot of variance of the target feature grouped by the categorical features. This plot gives us more confidence on our conclusion from second plot. For example, land surface condition or roof type has no any outliers whereas plan surface configuration has somewhat outliers. This also make sense because plan surface configuration also has more number of category levels in the feature compared to categorical features.</li>
</ul>

<br>

<b>D.4) Binary features:</b>
![alt_text](/images/cat_feat_target_var.png)

More than 50% of our feature set is binary. Binary features provide many useful information about a bulding such as does a building has any secondary usage, does stones are used for superstructure, does timber is used for superstructure, etc. The plot above indicates the target feature variance grouped by the individual binary feature levels. We can see that has_superstructure_mud_mortar_brick and has_superstructure_rc_engineered are two one of the features with lowest target feature variance.


## E) Evaluation metrics:

<img src="https://render.githubusercontent.com/render/math?math=F_{micro} = \frac{2.P_{micro}.R_{micro}}{P_{micro}%2BR_{micro}}">

We are using F1 metric for model evaluation, but more specifically we are using micro F1 metric. There are three different types of F1 metrics: macro, micro, weighted. The idea is simple, if we use micro precision and recall in calculating F1 metric then it is micro F1 metric, similarly, if we use macro or weighted precision and recall in calculating F1 metric then it is macro or weighted F1 metric respectively.<br>

<img src="https://render.githubusercontent.com/render/math?math=Pr = \frac{TP}{TP%2BFP}">
<img src="https://render.githubusercontent.com/render/math?math=Re = \frac{TP}{TP%2BFN}">

Where, Pr is Precision, Re is Recall, TP indicates number of True Positive predictions, FP indicates number of False Positive predictions, FN indicates number of False Negative predictions.<br>

Considering d number of target classes,
<img src="https://render.githubusercontent.com/render/math?math=Pr_{micro} = \frac{TP_{c1}%2B...%2BTP{cd}}{(TP_{c1}%2B...%2BTP{cd})%2B(FP_{c1}%2B...%2BFP{cd})}">

<img src="https://render.githubusercontent.com/render/math?math=Re_{micro} = \frac{TP_{c1}%2B...%2BTP{cd}}{(TP_{c1}%2B...%2BTP{cd})%2B(FN_{c1}%2B...%2BFN{cd})}">



## F) Models:

| Model Description                                           | Train score  | CV score  | Test score  |
|-------------------------------------------------------------|--------------|-----------|-------------|
| Logistic reg + all categorical (OHE) + Binary features      | 0.5761       | 0.5761    | 0.5817      |
| Decision tree (numeric features) + logistic regression (categorical features)   | 0.6354       | 0.6348    | 0.6346      |
| Random forest (numeric features) + logistic regression (categorical features)   | 0.6656       | 0.6617    | 0.6577      |
| Random forest (numeric features) + logistic regression (categorical features) + Mean Target encoding   | 0.7022       | 0.7020    | 0.7029      |
| Random forest (numeric features) + logistic regression (categorical features) + Mean Target encoding + Feature interaction   | 0.7171       | 0.7137    | 0.7152      |

## G) Model description:

We have total of 38 features (building information for any sample).

### G.1) Categorical features:<br>
First we observe all the 8 numeric features are not necessarily numeric in nature. For example, "count of floor pre-earthquake" is the feature that can be considered as categorical feature instead of numeric. Similarly, features such as "count of families living in the building" and "geo level 1 id" contain small number of unique values that can be considered as the categorical levels of the features.<br>
Hence, in total we have 5 numeric features, 11 categorical features and 22 binary features.<br>
One Hot Encoding is a simple technique where we can convery any k level categorical featue into k-1 (or k) binary features. After one hot encoding we get 132 binary features that we can X_train_bin for training set and X_test_bin for testing set.

### G.2) Numeric features:<br>
After the last step we are left with only 5 numeric features which itself are not very informative and useful. But the feature interaction brings a lot of information into the model. But building feature interactions with numeric features themselves is not very useful. Hence, we create feature interactions of "geo_level_1_id", "geo_level_2_id", "geo_level_3_id" with all the other features.<br>
Note, three features "age", "height_percentage", and "area_percentage" are still numerical and needs to be converted into categorical. For this task we use KMeans clustering with the cluster number as 5 to discretize the feature values into 5 different clusters (bins).<br>
After the feature interaction step we are left with 105 feature interactions.<br>
But, this step make features non-numerical and machine learning algorithms understands only numeric values. To make all of the feature interactions numerical we use Mean Target Encoding technique that replaces every unique feature value with the mean of target feature value grouped by that particular category.<br>
Note: Extra care needs to be taken while performing this step as it can introduce lot of variance in our model and makes model overfitted to our training data set. We use excellent implementation of this technique from ![Oliver's Kaggle notebook: mean target encoding with smoothing](https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features).

### G.3) Feature selection:
With the feature engineering steps, where the number of features has grown from 38 features to 237 features, feature selection step becomes important. Reducing the number of features maintains the principal of parsimony as well as helps speeding up the process of model training.

For feature selection of the feature interacted features, we use Greedy Forward Feature selection technique which starts with no feature in bucket and keeps adding one feature at a time only if that feature improves a model performance on the validation set. The implementation idea is take from ![Abhishek Thakur's Approaching (Almost) Any Machine Learning Problem book](https://www.amazon.com/Approaching-Almost-Machine-Learning-Problem/dp/8269211508/ref=sr_1_1?crid=10DSYMCBNNOSZ&dchild=1&keywords=abhishek+thakur&qid=1611284211&sprefix=abhishek+%2Caps%2C179&sr=8-1).<br>
Note: This step is very time consuming as we implement this step for k time for k-fold cross-validation.<br>
After this step we end up having just 11 feature interaction from 105 features.

### G.4) Model building:
In the actual building, we have numeric feature interacted features and binary categorical features.<br>
It is important to understand which type of machine learning model performs better of the types features we have. The tree based models are well suited for the numeric features that exhibit categorical nature where as logistic regression is well suited for high dimmensional sparse representation data sets. <br>
Now, we first build a decision tree model on the numeric feature interacted features and combines the probabilistic output of the model with sparse binary feature set. This combined sparse feature set is then feed to the logistic regression model to output final predictions.

## H) Conclusion:
<ul>
  <li>From the evaluation metric we can say current model is 71.52% accurate on the test data. Hence, local government can make use of the important fetures to make regulatory decisions for the constructions of the building in selected area.</li>
  <li>In every selected feature interaction one of the three geo level id features exist. In particular geo level id 2 and geo level id 3 are important features. These two features gives granular information about the geological position of a building. Indicating some geological locations are safer than others.</li>
  <li>Some of the other important features are the foundation types and construction types. Indicating some types are safer to use than the others.</li>
  <li>Based on these informations, local government can take neccessary actions in the targetted area to prevent larger building damages in case future disasters.</li>
<ul>
