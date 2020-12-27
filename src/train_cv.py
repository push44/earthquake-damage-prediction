# Train a decision tree classifier only on the TARGET ENCODED FEATURE INTERACTIONS
# Use probabilistic output of this classifier as a input feature for the logistic regression model where
# the features are binary numeric features and ohe hot encoded categorical features

import model_dispatcher
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
import numpy as np
from scipy import sparse

pd.options.mode.chained_assignment = None


def run():
    # Read train data with folds
    df = pd.read_csv("../input/train_folds.csv")

    # Target encoded feature interaction dataframe
    df_feature_inter = pd.read_csv("../input/target_encoded_feature_interaction.csv")

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

    # Lists to record evaluation scores
    train_micro_f1_score = []
    valid_micro_f1_score = []

    # k fold cross-validation
    for fold in tqdm(range(10)):

        # Train and validation y series
        y_train = df_ohe[df_ohe["kfold"] != fold]["damage_grade"].reset_index(drop=True)
        y_valid = df_ohe[df_ohe["kfold"] == fold]["damage_grade"].reset_index(drop=True)

        # Train and validation indicator features, after OHE
        X_train_bin = df_ohe[df_ohe["kfold"] != fold][ohe_columns].reset_index(drop=True)
        X_valid_bin = df_ohe[df_ohe["kfold"] == fold][ohe_columns].reset_index(drop=True)

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

        X_train_num = df_feature_inter[df_feature_inter["kfold"] != fold].\
            drop(["building_id", "kfold", "damage_grade"], axis=1)[selected_feature_interaction]

        X_valid_num = df_feature_inter[df_feature_inter["kfold"] == fold].\
            drop(["building_id", "kfold", "damage_grade"], axis=1)[selected_feature_interaction]

        # First Tree based model
        clf1 = model_dispatcher.models["decision_tree_clf"]
        clf1.fit(X_train_num, y_train)

        # Predictors for final prediction with stacked tree based model results
        X_train = sparse.csr_matrix(np.hstack((X_train_bin, clf1.predict_proba(X_train_num))))
        X_valid = sparse.csr_matrix(np.hstack((X_valid_bin, clf1.predict_proba(X_valid_num))))

        # Final model for prediction
        clf2 = model_dispatcher.models["logistic_reg"]
        clf2.fit(X_train, y_train)

        # Recoding evaluation score
        train_micro_f1_score.append(metrics.f1_score(y_train, clf2.predict(X_train), average="micro"))
        valid_micro_f1_score.append(metrics.f1_score(y_valid, clf2.predict(X_valid), average="micro"))

    print(np.mean(train_micro_f1_score), np.mean(valid_micro_f1_score))


if __name__ == "__main__":
    run()
