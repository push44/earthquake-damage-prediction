# Train a decision tree classifier only on the numerical features (avoiding binary numeric features)
# Use probabilistic output of this classifier as a input feature for the logistic regression model where
# the features are binary numeric features and ohe hot encoded categorical features

import model_dispatcher
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
import numpy as np
from scipy import sparse

pd.options.mode.chained_assignment = None


def label_encoding(df_enc):
    """
    :param df_enc: DataFrame with kfold, building_id and damage_grade
    :type df_enc: pandas dataframe
    :return: Categorical label encoded dataframe
    :rtype: pandas dataframe
    """
    for col in df_enc.select_dtypes(include="object").columns:
        if col in ["kfold", "damage_grade", "building_id"]:
            continue

        df_enc[col], _ = pd.factorize(df_enc[col], sort=True)
    return df_enc


def run():
    # Read train data with folds
    df = pd.read_csv("../input/train_folds.csv")

    # Train encoded dataframe
    df_enc = pd.read_csv("../input/target_mean_encoded_df.csv")

    # All features to considered as categorical
    categorical_features = [col for col in df.columns if col not in ["kfold", "building_id",
                                                                     "damage_grade", "geo_level_2_id",
                                                                     "geo_level_3_id", "age",
                                                                     "area_percentage", "height_percentage"]]

    # Convert categorical feature dtype as object
    for col in categorical_features:
        df[col] = df[col].astype("object")

    # One hot encoded features
    df_ohe = pd.get_dummies(df, prefix_sep="_ohe_")

    # List of feature names after OHE
    ohe_columns = [col for col in df_ohe.columns if "_ohe_" in col]

    # Lists to records evaluation score
    train_micro_f1_score = []
    valid_micro_f1_score = []

    # k fold cross-validation
    for fold in tqdm(range(10)):

        # Train and validation y series
        y_train = df_ohe[df_ohe["kfold"] != fold]["damage_grade"]
        y_valid = df_ohe[df_ohe["kfold"] == fold]["damage_grade"]

        # Train and validation indicator features, after OHE
        X_train_bin = df_ohe[df_ohe["kfold"] != fold][ohe_columns]
        X_valid_bin = df_ohe[df_ohe["kfold"] == fold][ohe_columns]

        # Train and validation numeric features with mean target encoding
        # List of features after mean target encoding
        numeric_encoded_features = [col + "_enc" for col in df if col not in
                                    categorical_features + ["kfold", "building_id", "damage_grade"]]

        X_train_num = df_enc[df_enc["kfold"] != fold][numeric_encoded_features]
        X_valid_num = df_enc[df_enc["kfold"] == fold][numeric_encoded_features]

        # Model
        clf1 = model_dispatcher.models["decision_tree_reg"]
        clf1.fit(X_train_num, y_train)

        X_train = sparse.csr_matrix(np.hstack((X_train_bin, clf1.predict(X_train_num).reshape(-1, 1))))
        X_valid = sparse.csr_matrix(np.hstack((X_valid_bin, clf1.predict(X_valid_num).reshape(-1, 1))))

        clf2 = model_dispatcher.models["logistic_reg"]
        clf2.fit(X_train, y_train)

        # Recoding evaluation score
        train_micro_f1_score.append(metrics.f1_score(clf2.predict(X_train), y_train, average="micro"))
        valid_micro_f1_score.append(metrics.f1_score(clf2.predict(X_valid), y_valid, average="micro"))

    print(np.mean(train_micro_f1_score), np.mean(valid_micro_f1_score))


if __name__ == "__main__":
    run()
