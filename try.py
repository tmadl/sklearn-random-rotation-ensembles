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