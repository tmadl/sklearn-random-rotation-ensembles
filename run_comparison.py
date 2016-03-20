from uci_comparison import compare_estimators
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from rr_forest import RRForestClassifier
from rr_extra_forest import RRExtraTreesClassifier

estimators = {
              'RandomForest': RandomForestClassifier(n_estimators=20),
              'RndRotForest': RRForestClassifier(n_estimators=20),
              'ExtraTrees': ExtraTreesClassifier(n_estimators=20),
              'RndRotETrees': RRExtraTreesClassifier(n_estimators=20),
            }

# optionally, pass a list of UCI dataset identifiers as the datasets parameter, e.g. datasets=['iris', 'diabetes']
# optionally, pass a dict of scoring functions as the metric parameter, e.g. metrics={'F1-score': f1_score}
compare_estimators(estimators)