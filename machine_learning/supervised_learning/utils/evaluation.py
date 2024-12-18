from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score,
    accuracy_score,
)
import numpy as np
import pandas as pd


def model_evaluation(
    y_pred: np.ndarray,
    y_true: pd.Series,
    y_pred_probability: np.ndarray,
    labels: list=None,
    toPrint: bool = True,
):

    """
    y_pred: Model's output as predicated labels
    y_true: Actual labels
    y_pred_probability: Predicated probability for the positive class
    toPrint: If True it will print all the results. By default it is True.
    """

    report = classification_report(
        y_pred=y_pred, y_true=y_true, labels=labels, digits=3, zero_division=1
    )
    roc_auc = roc_auc_score(
        y_true=y_true, y_score=y_pred_probability, labels=labels, average="weighted", multi_class="ovr"
    )
    f1_score_per_class = f1_score(
        y_true=y_true, y_pred=y_pred, labels=labels, average=None, zero_division=0
    )
    f1_score_avg = f1_score(
        y_true=y_true, y_pred=y_pred, labels=labels, average="weighted", zero_division=0
    )

    acc = accuracy_score(y_true=y_true, y_pred=y_pred,normalize=True)

    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)

    if toPrint:
        print(conf_matrix)
        print(report)
        print(f"ROC_AUC: {roc_auc * 100:.2f}")
        print("f1_score_per_class: ", f1_score_per_class)
        print("f1_score_avg: ", f1_score_avg)
        print("accuracy: ", acc)

    return roc_auc, f1_score_per_class, f1_score_avg, acc, report, conf_matrix
