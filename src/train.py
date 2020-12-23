import pandas as pd
import model_dispatcher
import pickle
from scipy import sparse
import numpy as np


def run():
    # Read train data with folds
    df = pd.read_csv("../input/train_folds.csv")

    # List binary columns
    binary_columns = []
    for col in df.select_dtypes(exclude="object").columns:
        if len(df[col].unique()) == 2 and col not in ["kfold", "damage_grade", "building_id"]:
            binary_columns.append(col)

    # One hot encoded features
    df = pd.get_dummies(df, prefix_sep="_ohe_")
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
