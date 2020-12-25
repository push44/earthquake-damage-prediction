import pandas as pd
import model_dispatcher
import pickle
from scipy import sparse
import numpy as np
import target_encoding

pd.options.mode.chained_assignment = None


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

    # Train encoded dataframe
    df_enc = pd.read_csv("../input/target_mean_encoded_df.csv")

    # Read test data and get it merged with submission file
    test_values = pd.read_csv("../input/test_values.csv")
    sub_df = pd.read_csv("../input/submission_format.csv").drop(["damage_grade"], axis=1)
    test_values = pd.merge(sub_df, test_values, on="building_id", how="right")

    # Note down number of train rows
    n_rows = train_values.shape[0] - 1

    # Concat train and test dataset for OHE
    df_temp = pd.concat((train_values, test_values), axis=0, ignore_index=True)

    # List of features to be considered as categorical
    categorical_features = [col for col in df_temp.columns if col not in ["kfold", "building_id", "damage_grade",
                                                                          "geo_level_2_id", "geo_level_3_id",
                                                                          "age", "area_percentage",
                                                                          "height_percentage"]]

    # Convert categorical feature dtype as object
    for col in categorical_features:
        df_temp[col] = df_temp[col].astype("object")

    # One hot encoded features
    df_ohe = pd.get_dummies(df_temp, prefix_sep="_ohe_")

    # List of feature names after OHE
    ohe_columns = [col for col in df_ohe.columns if "_ohe_" in col]

    # Use only train dataset
    df_train_ohe = df_ohe.loc[:n_rows, :]
    df_test_ohe = df_ohe.loc[n_rows+1:, :]

    # y series
    y_train = df_train_ohe["damage_grade"]

    # Train and test indicator features after OHE
    X_train_bin = df_train_ohe[ohe_columns]
    X_test_bin = df_test_ohe[ohe_columns]

    # Train and test numerical features with mean target encoding
    # List of features after mean target encoding
    numeric_encoded_features = [col + "_enc" for col in df_temp if col not in
                                categorical_features + ["kfold", "building_id", "damage_grade"]]

    for col in list(map(lambda val: val.split("_enc")[0], numeric_encoded_features)):
        _, df_test_ohe[col + "_enc"] = target_encoding.target_encode(trn_series=df_train_ohe[col],
                                                                     tst_series=df_test_ohe[col], target=y_train,
                                                                     min_samples_leaf=20, noise_level=0.08)

    X_train_num = df_enc[numeric_encoded_features]
    X_test_num = df_test_ohe[numeric_encoded_features]

    # Train model
    train(X_train_num, X_train_bin, y_train)

    # Prediction
    prediction_values = predict(X_test_num, X_test_bin)

    # Submission
    sub_df["damage_grade"] = prediction_values
    sub_df.to_csv("../input/submission_format.csv", index=False)


if __name__ == "__main__":
    run()
