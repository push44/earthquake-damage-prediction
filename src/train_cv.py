# Train a decision tree classifier only on the numerical features (avoiding binary numeric features)
# Use probabilistic output of this classifier as a input feature for the logistic regression model where
# the features are binary numeric features and ohe hot encoded categorical features

import model_dispatcher
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
import numpy as np
from scipy import sparse
from sklearn import preprocessing

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

    # List binary columns
    binary_columns = []
    for col in df.select_dtypes(exclude="object").columns:
        if len(df[col].unique()) == 2 and col not in ["kfold", "damage_grade", "building_id"]:
            binary_columns.append(col)

    # Convert geo_level_1_id as object
    df["geo_level_1_id"] = df["geo_level_1_id"].astype("object")
    df["count_floors_pre_eq"] = df["count_floors_pre_eq"].astype("object")
    df["count_families"] = df["count_families"].astype("object")

    # One hot encoded features
    df = pd.get_dummies(df, prefix_sep="_ohe_")

    columns = [col for col in df.columns if "_ohe_" in col] + binary_columns

    # Lists to records evaluation score
    train_micro_f1_score = []
    valid_micro_f1_score = []

    # k fold cross-validation
    for fold in tqdm(range(10)):

        # take out 9 train and 1 cv out of the dataframe
        df_train = df[df["kfold"] != fold]
        df_valid = df[df["kfold"] == fold]

        # X y split
        X_train, y_train = df_train.drop(["damage_grade", "kfold", "building_id"], axis=1), df_train["damage_grade"]
        X_valid, y_valid = df_valid.drop(["damage_grade", "kfold", "building_id"], axis=1), df_valid["damage_grade"]

        # Split numerical and binary features
        X_train_num = X_train[[col for col in X_train.columns if col not in columns]]
        X_valid_num = X_valid[[col for col in X_valid.columns if col not in columns]]

        X_train_bin = X_train[columns]
        X_valid_bin = X_valid[columns]

        # Feature engineering (Numeric)
        #power_trans = preprocessing.PowerTransformer()
        #X_train_num = power_trans.fit_transform(X_train_num+1)
        #X_valid_num = power_trans.transform(X_valid_num + 1)

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
