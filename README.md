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

## Models:

Evaluation metrics = F1 micro average score

| Model Description                                           | Train score  | CV score  | Test score  |
|-------------------------------------------------------------|--------------|-----------|-------------|
| Logistic reg + all categorical (OHE) + Binary features      | 0.5761       | 0.5761    | 0.5817      |
| Decision tree (numeric features) + logistic regression (categorical features)   | 0.6354       | 0.6348    | 0.6346      |
| Random forest (numeric features) + logistic regression (categorical features)   | 0.6656       | 0.6617    | 0.6577      |
| Random forest (numeric features) + logistic regression (categorical features) + Mean Target encoding   | 0.7022       | 0.7020    | 0.7029      |
| Random forest (numeric features) + logistic regression (categorical features) + Mean Target encoding + Feature interaction   | 0.7171       | 0.7137    | 0.7152      |
