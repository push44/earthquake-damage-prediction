from sklearn import linear_model
from sklearn import tree

models = {"logistic_reg": linear_model.SGDClassifier(loss="log", max_iter=10000, n_jobs=-1, random_state=1),
          "decision_tree": tree.DecisionTreeClassifier(criterion="gini", max_depth=3)}
