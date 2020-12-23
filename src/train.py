import pandas as pd
import model_dispatcher
import pickle
from scipy import sparse
import numpy as np


def run():
    # Read train data with folds
    df = pd.read_csv("../input/train_folds.csv")
    test_values = pd.read_csv("../input/test_values.csv")

    n_rows = df.shape[0] - 1

    df_temp = pd.concat((df, test_values), axis=0, ignore_index=True)

    # List binary columns
    binary_columns = []
    for col in df_temp.select_dtypes(exclude="object").columns:
        if len(df_temp[col].unique()) == 2 and col not in ["kfold", "damage_grade", "building_id"]:
            binary_columns.append(col)

    # Convert geo_level_1_id as object
    df_temp["geo_level_1_id"] = df_temp["geo_level_1_id"].astype("object")
    df_temp["count_floors_pre_eq"] = df_temp["count_floors_pre_eq"].astype("object")
    df_temp["count_families"] = df_temp["count_families"].astype("object")

    # One hot encoded features
    df_temp = pd.get_dummies(df_temp, prefix_sep="_ohe_")

    df = df_temp.loc[:n_rows, :]

    columns = [col for col in df.columns if "_ohe_" in col] + binary_columns

    # X y split
    X, y = df.drop(["kfold", "damage_grade", "building_id"], axis=1), df["damage_grade"]

    # Separate numerical and binary features
    X_num = X[[col for col in X.columns if col not in columns]]
    X_bin = X[columns]

    # Train model
    clf1 = model_dispatcher.models["decision_tree_reg"]
    clf1.fit(X_num, y)

    X_comb = sparse.csr_matrix(np.hstack((X_bin, clf1.predict(X_num).reshape(-1, 1))))

    clf2 = model_dispatcher.models["logistic_reg"]
    clf2.fit(X_comb, y)

    with open("../models/decision_tree_reg.pickle", "wb") as f:
        pickle.dump(clf1, f)

    with open("../models/binary_logistic_reg.pickle", "wb") as f:
        pickle.dump(clf2, f)


if __name__ == "__main__":
    run()
