from uci_comparison import compare_estimators
from sklearn.ensemble.forest import RandomForestClassifier
from rr_forest import RRForestClassifier

n_estimators = 160

estimators = {
              'RandomForest': RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1),
              'RndRotForest': RRForestClassifier(n_estimators=n_estimators, n_jobs=-1)
            }

results = compare_estimators(estimators)