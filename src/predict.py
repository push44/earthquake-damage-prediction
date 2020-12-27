import pandas as pd
import model_dispatcher
import pickle
from scipy import sparse
import numpy as np

pd.options.mode.chained_assignment = None


def train(X_num, X_bin, y):
    # Train model
    clf1 = model_dispatcher.models["decision_tree_clf"]
    clf1.fit(X_num, y)

    X_comb = sparse.csr_matrix(np.hstack((X_bin, clf1.predict_proba(X_num))))

    clf2 = model_dispatcher.models["logistic_reg"]
    clf2.fit(X_comb, y)

    with open("../models/decision_tree_clf.pickle", "wb") as f:
        pickle.dump(clf1, f)

    with open("../models/binary_logistic_reg.pickle", "wb") as f:
        pickle.dump(clf2, f)


def predict(X_num, X_bin):
    # Predict
    with open("../models/decision_tree_clf.pickle", "rb") as f:
        clf1 = pickle.load(f)

    with open("../models/binary_logistic_reg.pickle", "rb") as f:
        clf2 = pickle.load(f)

    X_comb = sparse.csr_matrix(np.hstack((X_bin, clf1.predict_proba(X_num))))

    return list(map(lambda val: int(val), clf2.predict(X_comb)))


def run():
    # Read train data with folds
    train_values = pd.read_csv("../input/train_folds.csv")

    # Read test data and get it merged with submission file
    test_values = pd.read_csv("../input/test_values.csv")
    sub_df = pd.read_csv("../input/submission_format.csv").drop(["damage_grade"], axis=1)
    test_values = pd.merge(sub_df, test_values, on="building_id", how="right")

    # Target encoded feature interaction
    train_feat_inter = pd.read_csv("../input/train_target_encoded_feature_interaction.csv")
    test_feat_inter = pd.read_csv("../input/test_target_encoded_feature_interaction.csv")

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

    # Train and validation target encoded feature interaction
    selected_feature_interaction = ['geo_level_2_id_ground_floor_type',
                                    'geo_level_3_id_has_superstructure_mud_mortar_stone',
                                    'geo_level_2_id_roof_type',
                                    'geo_level_3_id_foundation_type',
                                    'geo_level_2_id_has_superstructure_mud_mortar_stone',
                                    'geo_level_2_id_foundation_type',
                                    'geo_level_3_id_has_superstructure_adobe_mud',
                                    'geo_level_3_id_has_secondary_use_use_police',
                                    'geo_level_3_id_has_superstructure_rc_non_engineered',
                                    'geo_level_2_id_has_superstructure_cement_mortar_brick',
                                    'geo_level_3_id_ground_floor_type']

    X_train_num = train_feat_inter.drop(["building_id", "kfold", "damage_grade"], axis=1)[selected_feature_interaction]
    X_test_num = test_feat_inter.drop(["building_id"], axis=1)[selected_feature_interaction]

    # Train model
    train(X_train_num, X_train_bin, y_train)

    # Prediction
    prediction_values = predict(X_test_num, X_test_bin)

    # Submission
    sub_df["damage_grade"] = prediction_values
    sub_df.to_csv("../input/submission_format.csv", index=False)


if __name__ == "__main__":
    run()
