# Create

import pandas as pd
import target_encoding
from tqdm import tqdm
from sklearn import preprocessing

pd.options.mode.chained_assignment = None


def mean_target_encoding_val(data, features, target, k_fold):
    """
    :param data: Pandas dataframe with all columns
    :param features:  List of feature names to be encoded
    :param target: Target feature name
    :param k_fold: Integer values indicating k
    :return: Dataframe with mean target encoding for the specified features
    """
    # List to store k validation datasets
    encoded_dfs = []
    for fold in tqdm(range(k_fold)):
        # Split train and validation datasets
        df_train = data[data["kfold"] != fold]
        df_valid = data[data["kfold"] == fold]

        # Mean target encoded all columns from the feature set
        for column in features:
            _, valid_arr = target_encoding.target_encode(trn_series=df_train[column],
                                                         tst_series=df_valid[column],
                                                         target=df_train[target],
                                                         min_samples_leaf=20,
                                                         noise_level=0.08)
            df_valid.loc[:, column] = valid_arr

        # append to our list of encoded validation dataframes
        encoded_dfs.append(df_valid)
    # Concatenate validation dataframes
    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df


def run(tr_df, te_df):
    # List of features to be considered as numeric
    numeric_features = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]

    # List of features to be considered as categorical
    categorical_features = [col for col in tr_df.columns if col not in numeric_features + ["building_id",
                                                                                           "damage_grade", "kfold"]]

    for col in ["age", "area_percentage", "height_percentage"]:
        kbins = preprocessing.KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="kmeans")
        tr_df[col] = kbins.fit_transform(tr_df[col].values.reshape(-1, 1)).reshape(1, -1)[0]
        te_df[col] = kbins.transform(te_df[col].values.reshape(-1, 1)).reshape(1, -1)[0]

    # Convert all predictors' dtype as str for feature interaction
    for col in tr_df.columns:
        if col in ["building_id", "kfold", "damage_grade"]:
            continue
        tr_df[col] = tr_df[col].astype("str")
        te_df[col] = te_df[col].astype("str")

    # Create a dictionary to create feature interaction dataframe
    tr_feature_interaction_dict = {"building_id": tr_df["building_id"].values, "kfold": tr_df["kfold"].values,
                                   "damage_grade": tr_df["damage_grade"].values}

    te_feature_interaction_dict = {"building_id": te_df["building_id"].values}

    # Create feature interaction (all possible pairs of numeric predictors with categorical predictors)
    for num_col in tqdm(numeric_features):
        for cat_col in categorical_features:
            tr_feature_interaction_dict[num_col + "_" + cat_col] = tr_df[num_col] + "_" + tr_df[cat_col]
            te_feature_interaction_dict[num_col + "_" + cat_col] = te_df[num_col] + "_" + te_df[cat_col]

    # Obtain feature interaction dataframe
    tr_feature_interaction_df = pd.DataFrame(tr_feature_interaction_dict)
    te_feature_interaction_df = pd.DataFrame(te_feature_interaction_dict)

    # List all predictors as features for mean target encoding
    features = [col for col in tr_feature_interaction_df.columns if col not in ["kfold", "damage_grade", "building_id"]]

    for column in tqdm(features):
        _, test_arr = target_encoding.target_encode(trn_series=tr_feature_interaction_df[column],
                                                    tst_series=te_feature_interaction_df[column],
                                                    target=tr_feature_interaction_df["damage_grade"],
                                                    min_samples_leaf=20,
                                                    noise_level=0.08)
        te_feature_interaction_df.loc[:, column] = test_arr

    tr_feature_interaction_df = mean_target_encoding_val(data=tr_feature_interaction_df, features=features,
                                                         target="damage_grade", k_fold=10)

    # Obtain mean target encoding where values are obtained from validation set
    return tr_feature_interaction_df, te_feature_interaction_df


if __name__ == "__main__":
    # Read train data file
    train_values = pd.read_csv("../input/train_folds.csv")

    # Read test data file
    test_values = pd.read_csv("../input/test_values.csv")

    # Obtain mean target encoded feature interaction dataframe
    train_target_encoded_feature_interaction, test_target_encoded_feature_interaction = run(train_values, test_values)

    # Save target encoded feature interaction dataframes
    train_target_encoded_feature_interaction.to_csv("../input/train_target_encoded_feature_interaction.csv", index=False)
    test_target_encoded_feature_interaction.to_csv("../input/test_target_encoded_feature_interaction.csv", index=False)
