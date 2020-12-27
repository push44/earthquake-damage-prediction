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
    run(data)
