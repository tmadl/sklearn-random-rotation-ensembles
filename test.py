from rr_forest import RRForestClassifier
from uci_loader import getdataset
from sklearn.cross_validation import cross_val_score
X, y = getdataset('iris')
print cross_val_score(RRForestClassifier(n_estimators=20), X, y)