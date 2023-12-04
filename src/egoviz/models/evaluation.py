"""This module contains functions for evaluating sklearn models."""

import warnings
from typing import Optional, Protocol
import logging

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Classifier(Protocol):
    """Protocol for sklearn classifiers."""

    classes_: list

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


def logocv(df, X, y, groups, clf: Classifier):
    """Evaluate a classifier using leave one group out cross-validation."""

    logo = LeaveOneGroupOut()
    evaluation_metrics = {}
    all_y_true = []
    all_y_pred = []

    for train_index, test_index in logo.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # get key as group left out
        group_left_out = df.iloc[test_index]["video"].values[0][:5]

        # initialize and train the classifier
        clf.fit(X_train, y_train)

        # make predictions and evaluate the model
        y_pred = clf.predict(X_test)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            precision = precision_score(
                y_test, y_pred, average="macro", zero_division=1
            )
            recall = recall_score(y_test, y_pred, average="macro", zero_division=1)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=1)
        accuracy = accuracy_score(y_test, y_pred)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        # save results in a dict
        evaluation_metrics[group_left_out] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # get confusion matrix and results
    cm = confusion_matrix(all_y_true, all_y_pred)
    results = pd.DataFrame.from_dict(evaluation_metrics, orient="index")
    results = results.reset_index()
    results = results.rename(columns={"index": "group_left_out"})

    # get the median of the results
    results["median_accuracy"] = results["accuracy"].median()
    results["median_precision"] = results["precision"].median()
    results["median_recall"] = results["recall"].median()
    results["median_f1"] = results["f1"].median()

    # add model name
    results["model"] = clf.__class__.__name__

    # log complete
    logging.info("LOGOCV complete for %s", clf.__class__.__name__)

    return results, cm


def evaluate_models(models, df, label_encoder):
    """Evaluate a list of models using LOGOCV."""
    X = df.drop(["adl", "video"], axis=1)
    y = df["adl"]
    y_encoded = label_encoder.fit_transform(y)
    groups = df["video"].str[:5]

    results = {}

    for name, clf in models:
        result = logocv(df, X, y_encoded, groups, clf)
        results[name] = result

    # concat result[0] for result in results.values()
    results_df = pd.concat([result[0] for result in results.values()])

    return results, results_df
