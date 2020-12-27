import pickle
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True, 'font.size': 8})


def run(list_of_features):
    cnt_dict = Counter(list_of_features)
    x, h = zip(*sorted(list(zip(cnt_dict.keys(), cnt_dict.values())), key=lambda val: val[1], reverse=True))

    fig, ax = plt.subplots()
    fig.tight_layout()
    print(list(x))
    ax.barh(x, h)
    plt.savefig("../input/logistic_reg_feat_sel.png")
    plt.show()


if __name__ == "__main__":
    # Read pickle file from input
    with open("../input/dtc_feature_selection.pickle", "rb") as f:
        data = pickle.load(f)
        data = list(map(lambda fold_val: list(map(lambda val: val[0], fold_val)), data))

    feature_ind = []
    for fold_val in data:
        for ind in fold_val:
            feature_ind.append(ind)

    feat_inter_df = pd.read_csv("../input/target_encoded_feature_interaction.csv")
    train_values = pd.read_csv("../input/train_values.csv")

    numeric_columns = ["geo_level_2_id", "geo_level_3_id", "age", "area_percentage", "height_percentage"]

    df = pd.concat((feat_inter_df, train_values[numeric_columns]), axis=1)

    feature_dict = dict((ind, val) for ind, val in enumerate(list(df.columns)))

    feature_names_freq = []

    for feat in feature_ind:
        feature_names_freq.append(feature_dict[feat])

    run(feature_names_freq)
