import numpy as np
from uci_loader import getdataset

comparison_datasets = [
        "breast-cancer",
        "datasets-UCI breast-w",
        "datasets-UCI colic",
        "datasets-UCI credit-a",
        "datasets-UCI credit-g",
        "datasets-UCI diabetes",
        "uci-20070111 haberman",
        "heart",
        "ionosphere",
        "uci-20070111 labor",
        "liver-disorders",
        "mushrooms",
        "sonar",
        "uci-20070111 tic-tac-toe",
        "datasets-UCI vote"
    ]

X, y = getdataset(comparison_datasets[-1])

print X
print y
try:
    print X.toarray()
except:
    pass