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
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_validate


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


def leave_one_group_out_cv(
    df, X, y, groups, clf: Classifier, model_save_path: Optional[str] = None, xgb=False
) -> pd.DataFrame:
    """Evaluate a classifier using leave one group out cross-validation."""

    logo = LeaveOneGroupOut()
    evaluation_metrics = {}
    all_y_true = []
    all_y_pred = []

    for train_index, test_index in logo.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        if xgb:
            y_train, y_test = y[train_index], y[test_index]
        else:
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # get key as group left out
        group_left_out = df.iloc[test_index]["video"].values[0][:5]

        # Initialize and train the classifier
        clf.fit(X_train, y_train)

        # log that training is complete
        logging.info(
            "Training complete for %s, group left out: %s",
            clf.__class__.__name__,
            group_left_out,
        )

        # Make predictions and evaluate the model
        y_pred = clf.predict(X_test)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            precision = precision_score(
                y_test, y_pred, average="macro", zero_division=1
            )
            recall = recall_score(y_test, y_pred, average="macro", zero_division=1)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=1)
        accuracy = accuracy_score(y_test, y_pred)

        # # Plot confusion matrix for each group left out
        # cm = confusion_matrix(y_test, y_pred)
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(
        #     cm,
        #     annot=True,
        #     fmt="d",
        #     cmap="Blues",
        #     xticklabels=clf.classes_,
        #     yticklabels=clf.classes_,
        # )
        # plt.title(f"Confusion Matrix - Group Left Out: {group_left_out}")
        # plt.xlabel("Predicted")
        # plt.ylabel("True")
        # plt.show()

        # Save true and predicted labels for final confusion matrix
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        # save results in a dict
        evaluation_metrics[group_left_out] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # Plot the final confusion matrix with all true and predicted labels
    final_cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        final_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=clf.classes_,
        yticklabels=clf.classes_,
    )
    plt.title("Final Confusion Matrix - All Groups")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Save the final model if a file path is provided
    if model_save_path:
        with open(model_save_path, "wb") as model_file:
            pickle.dump(clf, model_file)

    results = pd.DataFrame.from_dict(evaluation_metrics, orient="index")
    results = results.reset_index()
    results = results.rename(columns={"index": "group_left_out"})

    # get mean accuracy, precision, recall, and f1-score
    results["mean_accuracy"] = results["accuracy"].mean()
    results["mean_precision"] = results["precision"].mean()
    results["mean_recall"] = results["recall"].mean()
    results["mean_f1"] = results["f1"].mean()

    # log complete
    logging.info("LOGOCV complete for %s", clf.__class__.__name__)

    return results
