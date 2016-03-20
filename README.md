sklearn-compatible Random Rotation Ensembles
===============

Scikit-learn compatible implementations of the recent Random Rotation Ensemble idea of [Blaser & Fryzlewicz, 2016, JMLR](http://jmlr.org/papers/volume17/blaser16a/blaser16a.pdf). 

The authors show that random rotations of the feature space in the individual classifiers within the ensemble significantly improves ensemble diversity, and thus overall ensemble accuracy; especially for tree-based ensembles.

Two such tree-based models are implemented here, Random Forests and Extremely Randomized Tree classifiers, and compared below to scikit's standard implementations.

The comparison suite may itself be useful for prototyping and testing new machine learning models (see usage below). 

Usage
===============

Usage example of the Random Rotation (RR) Ensembles:

```python
from uci_loader import *
X, y = getdataset('diabetes')

from rr_forest import RRForestClassifier
from rr_extra_forest import RRExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier

classifier = RRForestClassifier(n_estimators=20)
classifier.fit(X[:len(y)/2], y[:len(y)/2])
print "Random Rotation Forest Accuracy:", np.mean(classifier.predict(X[len(y)/2:]) == y[len(y)/2:])

classifier = RRExtraTreesClassifier(n_estimators=20)
classifier.fit(X[:len(y)/2], y[:len(y)/2])
print "Random Rotation Extra Trees Accuracy:", np.mean(classifier.predict(X[len(y)/2:]) == y[len(y)/2:])

classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X[:len(y)/2], y[:len(y)/2])
print "Random Forest Accuracy:", np.mean(classifier.predict(X[len(y)/2:]) == y[len(y)/2:])
```

Usage example for the UCI comparison:

```python
from uci_comparison import compare_estimators
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from rr_forest import RRForestClassifier
from rr_extra_forest import RRExtraTreesClassifier

estimators = {
              'RandomForest': RandomForestClassifier(n_estimators=160, n_jobs=-1),
              'RndRotForest': RRForestClassifier(n_estimators=160, n_jobs=-1),
              'ExtraTrees': ExtraTreesClassifier(n_estimators=160, n_jobs=-1),
              'RndRotETrees': RRExtraTreesClassifier(n_estimators=160, n_jobs=-1),
            }

# optionally, pass a list of UCI dataset identifiers as the datasets parameter, e.g. datasets=['iris', 'diabetes']
# optionally, pass a dict of scoring functions as the metric parameter, e.g. metrics={'F1-score': f1_score}
compare_estimators(estimators)
"""
                                   RF F1             RF Acc.           LinSVC F1         LinSVC Acc.
====================================================================================================
        iris (n=150)    0.389 (SE=0.048)   *0.940 (SE=0.009)    0.374 (SE=0.046)    0.913 (SE=0.009)
    diabetes (n=768)   *0.803 (SE=0.005)   *0.749 (SE=0.006)    0.499 (SE=0.027)    0.527 (SE=0.015)
"""
```