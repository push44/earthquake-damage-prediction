import pandas as pd
import model_dispatcher
import pickle


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

    # Only categorical columns
    cat_columns = list(df.select_dtypes(include="object").columns)

    # Columns have binary representation
    columns = ["building_id"]+cat_columns+binary_columns

    # Dataframe of all listed columns
    df = df[columns]

    # One hot encoded features
    df = pd.get_dummies(df)

    # Predictors
    X = df.drop(["building_id"], axis=1)

    # Train model
    with open("../models/binary_logistic_reg.pickle", "rb") as f:
        clf = pickle.load(f)

    sub_df["damage_grade"] = clf.predict(X)
    sub_df.to_csv("../input/submission_format.csv", index=False)


if __name__ == "__main__":
    run()
