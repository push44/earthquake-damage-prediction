from sklearn import linear_model
from sklearn import tree

models = {"logistic_reg": linear_model.SGDClassifier(loss="log", max_iter=10000, n_jobs=-1, random_state=1),
          "decision_tree_reg": tree.DecisionTreeRegressor(criterion="mse", max_depth=7, min_samples_split=20)}
