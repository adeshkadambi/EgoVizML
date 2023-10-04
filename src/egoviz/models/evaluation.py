"""This module contains functions for evaluating sklearn models."""

from typing import Protocol

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, KFold


class Classifier(Protocol):
    """Protocol for sklearn classifiers."""

    def fit(self, X, y):
        ...

    def predict(self, X):
        ...

    def predict_proba(self, X):
        ...


def get_preds(clf: Classifier, X_test):
    """Get predictions from a classifier."""
    return clf.predict(X_test)


def preds_dataframe(X_test, y_test, y_pred):
    """Get predictions from a classifier."""
    df = X_test.copy()
    df["y_true"] = y_test
    df["y_pred"] = y_pred
    return df


def evaluate_model(clf: Classifier, X_test, y_test):
    """Evaluate a classifier."""
    y_pred = get_preds(clf, X_test)

    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    preds_df = preds_dataframe(X_test, y_test, y_pred)

    return report, matrix, preds_df


def evaluate_k_fold(clf: Classifier, X, y, k=5, seed=0):
    """Evaluate a classifier using k-fold cross validation."""

    kf = KFold(n_splits=k, random_state=seed, shuffle=True)

    # get f1, precision, and recall scores for each fold
    scoring = ["f1_macro", "precision_macro", "recall_macro"]
    scores = cross_validate(
        clf,
        X,
        y,
        scoring=scoring,
        cv=kf,
        return_train_score=True,
    )

    # print the mean of each score
    print(
        f"f1_macro: {scores['test_f1_macro'].mean()} +/- {scores['test_f1_macro'].std()}"
    )

    score_df = pd.DataFrame(scores)
    score_df["model"] = clf.__class__.__name__
    score_df["mean_f1_macro"] = round(score_df["test_f1_macro"].mean(), 2)
    score_df["std_f1_macro"] = round(score_df["test_f1_macro"].std(), 2)

    return score_df
