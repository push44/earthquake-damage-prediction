import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":
    df = pd.read_csv("../input/train_values.csv")
    y = pd.read_csv("../input/train_labels.csv")
    df = pd.merge(df, y, on="building_id", how="inner")

    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=10)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y["damage_grade"].values)):
        df.loc[v_, "kfold"] = f

    df.to_csv("../input/train_folds.csv", index=False)
