import numpy as np
from sklearn.metrics.classification import accuracy_score, f1_score
import re, string
from uci_loader import getdataset, tonumeric
from sklearn.cross_validation import KFold
from scipy.stats.stats import mannwhitneyu, ttest_ind

comparison_datasets = [
        "breast-cancer",
        "datasets-UCI breast-w",
        "datasets-UCI credit-g",
        "uci-20070111 haberman",
        "heart",
        "ionosphere",
        "uci-20070111 labor",
        "liver-disorders",
        "uci-20070111 tic-tac-toe",
        "datasets-UCI vote"
    ]

metrics = {
           #'Acc.': accuracy_score, 
           'F1score': f1_score
        }

def shorten(d):
    return "".join(re.findall("[^\W\d_]", d.lower().replace('datasets-', '').replace('uci', '')))

def print_results_table(results, rows, cols, cellsize=20):
    row_format =("{:>"+str(cellsize)+"}") * (len(cols) + 1)
    print row_format.format("", *cols)
    print "".join(["="]*cellsize*(len(cols)+1))
    for rh, row in zip(rows, results):
        print row_format.format(rh, *row)

def compare_estimators(estimators, datasets = comparison_datasets, metrics = metrics, n_cv_folds = 10, decimals = 3, cellsize = 22):
    if type(estimators) != dict:
        raise Exception("First argument needs to be a dict containing 'name': Estimator pairs")
    if type(metrics) != dict:
        raise Exception("Argument metrics needs to be a dict containing 'name': scoring function pairs")
    cols = []
    for e in range(len(estimators)):
        for mname in metrics.keys():
            cols.append(sorted(estimators.keys())[e]+" "+mname)
    
    rows = []
    mean_results = []
    std_results = []
    for d in datasets:
        print "comparing on dataset",d
        mean_result = []
        std_result = []
        X, y = getdataset(d)
        rows.append(shorten(d)+" (n="+str(len(y))+")")
        for e in range(len(estimators.keys())):
            est = estimators[sorted(estimators.keys())[e]]
            mresults = [[] for i in range(len(metrics))]
            for train_idx, test_idx in KFold(len(y), n_folds=n_cv_folds):
                est.fit(X[train_idx, :], y[train_idx])
                y_pred = est.predict(X[test_idx, :])
                for i in range(len(metrics)):
                    try:
                        mresults[i].append(metrics.values()[i](y[test_idx], y_pred))
                    except:
                        mresults[i].append(metrics.values()[i](tonumeric(y[test_idx]), tonumeric(y_pred)))

            for i in range(len(metrics)):
                mean_result.append(np.mean(mresults[i]))
                std_result.append(np.std(mresults[i])/n_cv_folds)
        mean_results.append(mean_result)
        std_results.append(std_result)
    
    results = []
    for i in range(len(datasets)):
        result = []
        
        sigstars = ["*"]*(len(estimators)*len(metrics))
        for j in range(len(estimators)):
            for k in range(len(metrics)):
                for l in range(len(estimators)):
                    #if j != l and mean_results[i][j*len(metrics)+k] < mean_results[i][l*len(metrics)+k] + 2*(std_results[i][j*len(metrics)+k] + std_results[i][l*len(metrics)+k]):
                    if j != l and mean_results[i][j*len(metrics)+k] < mean_results[i][l*len(metrics)+k]:
                        sigstars[j*len(metrics)+k] = ""
        
        for j in range(len(estimators)):
            for k in range(len(metrics)):
                result.append((sigstars[j*len(metrics)+k]+"%."+str(decimals)+"f (SE=%."+str(decimals)+"f)") % (mean_results[i][j*len(metrics)+k], std_results[i][j*len(metrics)+k]))
        results.append(result)

    print_results_table(results, rows, cols, cellsize)
        
    return mean_results, std_results, results
