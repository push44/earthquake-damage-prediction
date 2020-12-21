import pandas as pd
import model_dispatcher
import pickle


def run():
    # Read train data with folds
    df = pd.read_csv("../input/train_folds.csv")

    # List binary columns
    binary_columns = []
    for col in df.select_dtypes(exclude="object").columns:
        if len(df[col].unique()) == 2 and col not in ["kfold", "damage_grade", "building_id"]:
            binary_columns.append(col)

    # Only categorical columns
    cat_columns = list(df.select_dtypes(include="object").columns)

    # Columns have binary representation
    columns = ["damage_grade", "building_id"]+cat_columns+binary_columns

    # Dataframe of all listed columns
    df = df[columns]

    # One hot encoded features
    df = pd.get_dummies(df)

    # X y split
    X, y = df.drop(["damage_grade", "building_id"], axis=1), df["damage_grade"]

    # Train model
    clf = model_dispatcher.models["logistic_reg"]
    clf.fit(X, y)

    with open("../models/binary_logistic_reg.pickle", "wb") as f:
        pickle.dump(clf, f)


if __name__ == "__main__":
    run()
