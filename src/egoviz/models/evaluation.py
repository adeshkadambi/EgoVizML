"""This module contains functions for evaluating sklearn models."""

from typing import Protocol

from sklearn.metrics import classification_report, confusion_matrix


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


def evaluate_model(clf: Classifier, X_test, y_test):
    """Evaluate a classifier."""
    y_pred = get_preds(clf, X_test)

    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    return report, matrix
