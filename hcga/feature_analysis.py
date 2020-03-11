"""function for analysis of graph features"""

import time

import numpy as np
import pandas as pd
import shap
import xgboost
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import filter_features
from .plotting import *

def analysis(
    features,
    features_info,
    shap=True,
    classifier_type="RF",
    verbose=True,
    folder=".",
    kfold=True,
    plot=True,
):
    """main function to classify graphs and plot results"""

    good_features = filter_features(features)
    normed_features = normalise_feature_data(good_features)
    
    #if reduced_set:
        
    

    if classifier_type == "RF":
        from sklearn.ensemble import RandomForestClassifier

        classifier = RandomForestClassifier(n_estimators=100, max_depth=30)
    elif classifier_type == "LGBM":
        from lightgbm import LGBMClassifier

        classifier = LGBMClassifier()
    else:
        raise Exception("Unknown classifier type: {}".format(classifier_type))

    if kfold:
        X,  explainer, shap_values, top_features = fit_model_kfold(
            normed_features, classifier=classifier, verbose=verbose
        )
    else:
        X, explainer, shap_values, top_features = fit_model(
            normed_features, classifier=classifier, verbose=verbose
        )

    if plot:

        if shap:
            
            shap_plots(X, shap_values)
        else:
            basic_plots(X, top_features)


    # TODO: have a consistent ouptut of feature classification to save, and plot later
    return X, explainer, shap_values


def fit_model_kfold(features, compute_shap=True, classifier=None, verbose=False):
    """shapeley analysis"""

    if classifier is None:
        raise Exception("Please provide a model for classification")

    X, y = _features_to_Xy(features)

#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
#    )

    n_splits = _number_folds(y)
    print("Using", n_splits, "splits")

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    shap_values = None
    explainer = None


    top_features = []    
    acc_scores = []
    oof_preds = np.zeros(y.shape[0])
    for train_index, val_index in folds.split(X, y=y):

        X_train, X_val = X.loc[train_index], X.loc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        classifier.fit(
            X_train, y_train,
        )
        top_features.append(classifier.feature_importances_)

        
        if compute_shap:
            explainer = shap.TreeExplainer(
                classifier, feature_perturbation="interventional",
            )
    
            if shap_values is None:
                shap_values = explainer.shap_values(X,check_additivity=False)
            else:
                shap_values = [
                    x + y
                    for x, y in zip(
                        shap_values, explainer.shap_values(X,check_additivity=False)
                    )
                ]

        oof_preds[val_index] = classifier.predict(X_val)
        acc_scores.append(balanced_accuracy_score(y_val, oof_preds[val_index]))

        if verbose:
            print("Fold accuracy: --- {0:.3f} ---)".format(acc_scores[-1]))

    if verbose:
        print("Balanced accuracy: ", np.mean(acc_scores))

    if compute_shap:
        shap_values = [x / n_splits for x in shap_values]
        #shap.summary_plot(shap_values, X_test)

    return X, explainer, shap_values, top_features



def fit_model(features, compute_shap=True, classifier=None, verbose=False):
    """shapeley analysis"""

    explainer = None
    shap_values = None   
    
    if classifier is None:
        raise Exception("Please provide a model for classification")

    X, y = _features_to_Xy(features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )

    classifier.fit(X_train, y_train)
    top_features=classifier.feature_importances_

    if compute_shap:
        explainer = shap.TreeExplainer(classifier, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_test, check_additivity=False)
        
    acc_scores = balanced_accuracy_score(y_test, classifier.predict(X_test))

    if verbose:
        print("Balanced accuracy: ", acc_scores)

#    if compute_shap:
#        shap.summary_plot(shap_values[0], X_test)  # , plot_type='dot')
#        force = shap.force_plot(
#            explainer.expected_value[0], shap_values[0][0, :], X_test.iloc[0, :]
#        )
#        shap.save_html("test.html", force)
    # shap.dependence_plot("harmonic centrality_max_E", shap_values[0], X_test)

    return X, explainer, shap_values, top_features



def _features_to_Xy(features):
    """decompose features dataframe to X and y"""
    X = features.drop(columns=["labels"])
    y = features["labels"]
    return X, y


def normalise_feature_data(features):
    """Normalise the feature matrix using sklearn scaler to remove the mean and scale to unit variance"""
    labels = features["labels"]
    normed_features = pd.DataFrame(
        StandardScaler().fit_transform(features), columns=features.columns
    )
    normed_features["labels"] = labels
    return normed_features


def _number_folds(y):
    counts = np.bincount(y)
    n_splits = int(np.min(counts[counts > 0]) / 2)
    return np.clip(n_splits, 2, 10)









def reduce_feature_set(X, y, top_features, classifier,importance_threshold=0.9):
    """    Reduce the feature set   """

    
    mean_importance = np.mean(np.array(top_features), axis=0)
    rank_feat = np.argsort(mean_importance)[::-1]
    n_feat = len(
        np.where(
            np.cumsum(mean_importance[rank_feat])
            < importance_threshold * mean_importance.sum()
        )[0]
    )
    n_feat = max(3, n_feat)
    rank_feat = rank_feat[:n_feat]

    print(n_feat, "to get .9 of total importance")

    X_reduced = X.iloc[:, rank_feat]

    return X_reduced




