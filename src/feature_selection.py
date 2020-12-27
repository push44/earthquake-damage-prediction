import pandas as pd
from sklearn import metrics
import model_dispatcher
import pickle
from scipy import sparse
from tqdm import tqdm

pd.options.mode.chained_assignment = None


class GreedyFeatureSelection:
    def __init__(self, model):
        self.model = model

    def evaluate_score(self, X_train, X_valid, y_train, y_valid):
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_valid)
        return metrics.f1_score(y_valid, predictions, average="micro")

    def feature_selection(self, X_train, X_valid, y_train, y_valid):
        # Lists to record selected features and respective scores
        good_features = []
        best_scores = []

        # Note number of features
        num_features = X_train.shape[1]

        # Count for process monitor
        cnt = 1
        while True:
            print("Iter count:", cnt)

            # For every iteration set current feature none and best score 0
            this_feature = None
            best_score = 0

            # Iterate over all features except those are already selected
            for feature in range(num_features):
                if feature in good_features:
                    continue

                # Try every feature one by one and check model performance
                selected_features = good_features + [feature]

                # Obtain train and validation predictors
                Xtrain = X_train[:, selected_features]
                Xvalid = X_valid[:, selected_features]

                # Obtain the score
                score = self.evaluate_score(sparse.csr_matrix(Xtrain), sparse.csr_matrix(Xvalid), y_train, y_valid)

                if score > best_score:
                    # If score improved then current feature should be tested against upcoming list of features
                    # Hence, update this_feature and best_score
                    print("feature:", feature, "score:", score)
                    this_feature = feature
                    best_score = score

            if this_feature is None:
                pass
            else:
                # At the end of a loop if this_feature contains any feature then update then that feature is selected
                # Hence, update selected features (good_features) and main_model score (best_score)
                print("Best score:", best_score)
                good_features.append(this_feature)
                best_scores.append(best_score)

            if len(best_scores) > 2:
                # Set tolerance of improvement
                if best_scores[-1] - best_scores[-2] < 0.005:
                    break

            cnt += 1

        return best_scores[:-1], good_features[:-1]


if __name__ == "__main__":
    # Read target encoded feature interaction dataframe
    feat_inter_df = pd.read_csv("../input/target_encoded_feature_interaction.csv")

    # Read train dataframe
    train_values = pd.read_csv("../input/train_values.csv")

    # Numeric features
    numeric_columns = ["geo_level_2_id", "geo_level_3_id", "age", "area_percentage", "height_percentage"]

    # Combine all numeric and target encoded feature interaction
    df = pd.concat((feat_inter_df, train_values[numeric_columns]), axis=1)

    # List to record each fold output
    fold_feat = []
    for fold in tqdm(range(10)):

        # Train and validation predictors
        X_train = df[df["kfold"] != fold].drop(["building_id", "kfold", "damage_grade"], axis=1)
        X_valid = df[df["kfold"] == fold].drop(["building_id", "kfold", "damage_grade"], axis=1)

        # List of all features
        all_features = X_train.columns

        # Train and validation target features
        y_train = feat_inter_df[feat_inter_df["kfold"] != fold]["damage_grade"]
        y_valid = feat_inter_df[feat_inter_df["kfold"] == fold]["damage_grade"]

        # Classifier to use for feature selection
        clf = model_dispatcher.models["decision_tree_clf"]

        # Feature selection
        feat_sel = GreedyFeatureSelection(clf)
        scores, features = feat_sel.feature_selection(X_train.values, X_valid.values, y_train, y_valid)

        # Record output
        fold_feat.append(list(zip(features, scores)))

    # Pickle list of fold outputs
    with open("../input/dtc_feature_selection.pickle", "wb") as f:
        pickle.dump(fold_feat, f)
