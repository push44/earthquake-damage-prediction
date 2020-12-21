import model_dispatcher
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
import numpy as np


def label_encoding(df_enc):
    """
    :param df_enc: DataFrame with kfold and damage_grade
    :type df_enc: pandas dataframe
    :return: Categorical label encoded dataframe
    :rtype: pandas dataframe
    """
    for col in df_enc.columns:
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

    # Only categorical columns
    cat_columns = list(df.select_dtypes(include="object").columns)

    # Columns have binary representation
    columns = ["kfold", "damage_grade", "building_id"]+cat_columns+binary_columns

    # Dataframe of all listed columns
    df = df[columns]

    # One hot encoded features
    df = pd.get_dummies(df)

    # Lists to records evaluation score
    train_micro_f1_score = []
    cv_micro_f1_score = []

    # k fold cross-validation
    for fold in tqdm(range(10)):

        # take out 9 train and 1 cv out of the dataframe
        df_train = df[df["kfold"] != fold]
        df_cv = df[df["kfold"] == fold]

        # X y split
        X_train, y_train = df_train.drop(["damage_grade", "kfold", "building_id"], axis=1), df_train["damage_grade"]
        X_cv, y_cv = df_cv.drop(["damage_grade", "kfold", "building_id"], axis=1), df_cv["damage_grade"]

        # Model
        clf = model_dispatcher.models["logistic_reg"]
        clf.fit(X_train, y_train)

        # Recoding evaluation score
        train_micro_f1_score.append(metrics.f1_score(clf.predict(X_train), y_train, average="micro"))
        cv_micro_f1_score.append(metrics.f1_score(clf.predict(X_cv), y_cv, average="micro"))

    print(np.mean(train_micro_f1_score), np.mean(cv_micro_f1_score))


if __name__ == "__main__":
    run()
