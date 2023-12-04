"""This module contains functions for visualizing sklearn models or images."""

import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from egoviz.models.evaluation import Classifier


def plot_cm(cm, clf: Classifier, title: str = "Confusion Matrix", figsize=(8, 6)):
    """Plot a confusion matrix."""
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=clf.classes_,
        yticklabels=clf.classes_,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def draw_boxes(
    img_path: str,
    boxes: list[list[int]],
    labels: list[str],
    active: list[bool],
    save_path: str | None = None,
):
    """Draw bounding boxes on an image."""

    # read image
    img = plt.imread(img_path)

    # draw boxes
    for box, label, act in zip(boxes, labels, active):
        color = (0, 255, 0) if act else (255, 0, 0)
        img = draw_box(img, box, label, color=color)

    # save image
    if save_path:
        plt.imsave(save_path, img)

    # show image without axis
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def draw_box(img, box, label, color=(0, 255, 0)):
    """Draw a single bounding box on an image."""
    x1, y1, x2, y2 = box
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)
    img = cv2.putText(
        img,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color=color,
        thickness=2,
    )
    return img
