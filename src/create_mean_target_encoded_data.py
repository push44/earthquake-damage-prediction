import pandas as pd
import target_encoding
from tqdm import tqdm
pd.options.mode.chained_assignment = None


def mean_target_encoding(data, features, target, k_fold):
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
            df_valid.loc[:, column + "_enc"] = valid_arr

        # append to our list of encoded validation dataframes
        encoded_dfs.append(df_valid)
    # Concatenate validation dataframes
    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df


# Read train dataframe
df = pd.read_csv("../input/train_folds.csv")

# List of features to be considered as categorical
categorical_features = [col for col in df.columns if col not in ["kfold", "building_id", "damage_grade"]]

# Convert categorical features' dtype as object
for col in categorical_features:
    df[col] = df[col].astype("object")

df = mean_target_encoding(data=df, features=categorical_features, target="damage_grade", k_fold=10)
df.to_csv("../input/target_mean_encoded_df.csv", index=False)
