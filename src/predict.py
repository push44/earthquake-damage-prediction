import pandas as pd
import pickle
from scipy import sparse
import numpy as np


def run():
    # Read train data with folds
    df = pd.read_csv("../input/test_values.csv")
    sub_df = pd.read_csv("../input/submission_format.csv").drop(["damage_grade"], axis=1)
    df = pd.merge(sub_df, df, on="building_id", how="right")
    train_values = pd.read_csv("../input/train_values.csv")

    n_rows = train_values.shape[0]

    df_temp = pd.concat((train_values, df), axis=0, ignore_index=True)

    # List binary columns
    binary_columns = []
    for col in df_temp.select_dtypes(exclude="object").columns:
        if len(df_temp[col].unique()) == 2 and col not in ["building_id"]:
            binary_columns.append(col)

    # Convert geo_level_1_id as object
    df_temp["geo_level_1_id"] = df_temp["geo_level_1_id"].astype("object")
    df_temp["count_floors_pre_eq"] = df_temp["count_floors_pre_eq"].astype("object")
    df_temp["count_families"] = df_temp["count_families"].astype("object")

    # One hot encoded features
    df_temp = pd.get_dummies(df_temp, prefix_sep="_ohe_")

    df = df_temp.loc[n_rows:, :]

    columns = [col for col in df.columns if "_ohe_" in col] + binary_columns

    # Predictors
    X = df.drop(["building_id"], axis=1)

    # Separate numerical and binary features
    X_num = X[[col for col in X.columns if col not in columns]]
    X_bin = X[columns]

    # Predict
    with open("../models/decision_tree_reg.pickle", "rb") as f:
        clf1 = pickle.load(f)

    with open("../models/binary_logistic_reg.pickle", "rb") as f:
        clf2 = pickle.load(f)

    X_comb = sparse.csr_matrix(np.hstack((X_bin, clf1.predict(X_num).reshape(-1, 1))))

    sub_df["damage_grade"] = list(map(lambda val: int(val), clf2.predict(X_comb)))
    sub_df.to_csv("../input/submission_format.csv", index=False)


if __name__ == "__main__":
    run()
