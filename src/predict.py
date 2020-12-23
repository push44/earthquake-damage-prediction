import pandas as pd
import model_dispatcher
import pickle
from scipy import sparse
import numpy as np
import target_encoding


def train(X_num, X_bin, y):
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


def predict(X_num, X_bin):
    # Predict
    with open("../models/decision_tree_reg.pickle", "rb") as f:
        clf1 = pickle.load(f)

    with open("../models/binary_logistic_reg.pickle", "rb") as f:
        clf2 = pickle.load(f)

    X_comb = sparse.csr_matrix(np.hstack((X_bin, clf1.predict(X_num).reshape(-1, 1))))

    return list(map(lambda val: int(val), clf2.predict(X_comb)))


def run():
    # Read train data with folds
    train_values = pd.read_csv("../input/train_folds.csv")
    test_values = pd.read_csv("../input/test_values.csv")
    sub_df = pd.read_csv("../input/submission_format.csv").drop(["damage_grade"], axis=1)

    test_values = pd.merge(sub_df, test_values, on="building_id", how="right")

    # Note down number of train rows
    n_rows = train_values.shape[0] - 1

    # Concat train and test dataset for OHE
    df_temp = pd.concat((train_values, test_values), axis=0, ignore_index=True)

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

    # Use only train dataset
    df_train = df_temp.loc[:n_rows, :]
    df_test = df_temp.loc[n_rows+1:, :]

    # Columns to avoid being considered as numeric
    columns = [col for col in df_train.columns if "_ohe_" in col] + binary_columns

    # X y split
    X_train, y_train = df_train.drop(["kfold", "damage_grade", "building_id"], axis=1), df_train["damage_grade"]
    X_test, y_test = df_test.drop(["kfold", "damage_grade", "building_id"], axis=1), df_test["damage_grade"]

    # Separate numerical and binary features
    X_train_num = X_train[[col for col in X_train.columns if col not in columns]]
    X_train_bin = X_train[columns]

    X_test_num = X_test[[col for col in X_train.columns if col not in columns]]
    X_test_bin = X_test[columns]

    # Feature engineering
    train_encoding = []
    test_encoding = []
    for col in X_train_num.columns:
        train_arr, test_arr = target_encoding.target_encode(trn_series=X_train_num[col],
                                                            tst_series=X_test_num[col],
                                                            target=y_train)
        train_encoding.append(train_arr)
        test_encoding.append(test_arr)

    X_train_num = np.array(train_encoding).T
    X_test_num = np.array(test_encoding).T

    # Train model
    train(X_train_num, X_train_bin, y_train)

    # Prediction
    prediction_values = predict(X_test_num, X_test_bin)

    # Submission
    sub_df["damage_grade"] = prediction_values
    sub_df.to_csv("../input/submission_format.csv", index=False)


if __name__ == "__main__":
    run()
