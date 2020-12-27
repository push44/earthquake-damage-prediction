from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
import xgboost as xgb

models = {"logistic_reg": linear_model.SGDClassifier(loss="log", max_iter=10000, n_jobs=-1, random_state=1),
          "decision_tree_reg": tree.DecisionTreeRegressor(criterion="mse", max_depth=7, min_samples_split=20),
          "decision_tree_clf": tree.DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_split=20),
          "random_forest_clf": ensemble.RandomForestClassifier(n_estimators=200, criterion="gini", max_depth=8,
                                                               min_samples_leaf=20, n_jobs=-1),
          "xgb_reg": xgb.XGBRegressor(n_estimators=100, max_depth=5)}
