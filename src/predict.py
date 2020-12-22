import pandas as pd
import pickle
from scipy import sparse
import numpy as np


def run():
    # Read train data with folds
    df = pd.read_csv("../input/test_values.csv")
    sub_df = pd.read_csv("../input/submission_format.csv").drop(["damage_grade"], axis=1)
    df = pd.merge(sub_df, df, on="building_id", how="right")

    # List binary columns
    binary_columns = []
    for col in df.select_dtypes(exclude="object").columns:
        if len(df[col].unique()) == 2 and col not in ["building_id"]:
            binary_columns.append(col)

    # One hot encoded features
    df = pd.get_dummies(df, prefix_sep="_ohe_")
    columns = [col for col in df.columns if "_ohe_" in col] + binary_columns

    # Predictors
    X = df.drop(["building_id"], axis=1)

    # Separate numerical and binary features
    X_num = X[[col for col in X.columns if col not in columns]]
    X_bin = X[columns]

    # Predict
    with open("../models/numeric_random_forest.pickle", "rb") as f:
        clf1 = pickle.load(f)

    with open("../models/binary_logistic_reg.pickle", "rb") as f:
        clf2 = pickle.load(f)

    X_comb = sparse.csr_matrix(np.hstack((X_bin, clf1.predict_proba(X_num))))

    sub_df["damage_grade"] = clf2.predict(X_comb)
    sub_df.to_csv("../input/submission_format.csv", index=False)


if __name__ == "__main__":
    run()
