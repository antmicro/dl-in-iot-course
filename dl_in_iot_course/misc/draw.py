"""
Methods for drawing plots.
"""

from matplotlib import pyplot as plt
from matplotlib import patheffects
from typing import List, Tuple, Optional
import numpy as np
import itertools
from pathlib import Path
from matplotlib import gridspec


def draw_confusion_matrix(
    confusion_matrix: np.ndarray,
    outpath: Optional[Path],
    title: str,
    class_names: List[str],
    cmap=None,
    figsize: Optional[Tuple] = None,
    dpi: Optional[int] = None,
):
    """
    Creates a confusion matrix plot.

    Parameters
    ----------
    confusion_matrix : ArrayLike
        Square numpy matrix containing the confusion matrix.
        0-th axis stands for ground truth, 1-st axis stands for predictions
    outpath : Optional[Path]
        Path where the plot will be saved. If None, the plot will be displayed.
    title : str
        Title of the plot
    class_names : List[str]
        List of the class names
    cmap : Any
        Color map for the plot
    figsize : Optional[Tuple]
        The size of the plot
    dpi : Optional[int]
        The dpi of the plot
    """
    if cmap is None:
        cmap = plt.get_cmap("BuPu")

    confusion_matrix = np.array(confusion_matrix, dtype=np.float32, copy=True)

    # compute sensitivity
    correctactual = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
    correctactual = correctactual.reshape(1, len(class_names))

    # compute precision
    correctpredicted = confusion_matrix.diagonal() / confusion_matrix.sum(axis=0)
    correctpredicted = correctpredicted.reshape(len(class_names), 1)

    # compute overall accuracy
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    # normalize confusion matrix
    confusion_matrix /= confusion_matrix.sum(axis=0)
    confusion_matrix = confusion_matrix.transpose()

    if figsize is None:
        figsize = [35, 35]

    if dpi is None:
        dpi = 216

    # create axes
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(len(class_names) + 1, len(class_names) + 1)
    axConfMatrix = fig.add_subplot(gs[0 : len(class_names), 0 : len(class_names)])
    axPredicted = fig.add_subplot(
        gs[len(class_names), 0 : len(class_names)], sharex=axConfMatrix
    )
    axActual = fig.add_subplot(
        gs[0 : len(class_names), len(class_names)], sharey=axConfMatrix
    )
    axTotal = fig.add_subplot(
        gs[len(class_names), len(class_names)], sharex=axActual, sharey=axPredicted
    )

    # define ticks for classes
    ticks = np.arange(len(class_names))

    # configure and draw confusion matrix
    axConfMatrix.set_xticks(ticks)
    axConfMatrix.set_xticklabels(class_names, fontsize="large", rotation=90)
    axConfMatrix.set_yticks(ticks)
    axConfMatrix.set_yticklabels(class_names, fontsize="large")
    axConfMatrix.set_xlabel("Actual class", fontsize="x-large")
    axConfMatrix.set_ylabel("Predicted class", fontsize="x-large")
    img = axConfMatrix.imshow(
        confusion_matrix,
        interpolation="nearest",
        cmap=cmap,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    axConfMatrix.xaxis.set_ticks_position("top")
    axConfMatrix.xaxis.set_label_position("top")

    # add percentages for confusion matrix
    for i, j in itertools.product(range(len(class_names)), range(len(class_names))):
        txt = axConfMatrix.text(
            j,
            i,
            (
                "100"
                if confusion_matrix[i, j] == 1.0
                else f"{100.0 * confusion_matrix[i, j]:3.1f}"
            ),
            ha="center",
            va="center",
            color="black",
            fontsize="medium",
        )
        txt.set_path_effects([patheffects.withStroke(linewidth=5, foreground="w")])

    # configure and draw sensitivity percentages
    axPredicted.set_xticks(ticks)
    axPredicted.set_yticks([0])
    axPredicted.set_xlabel("Sensitivity", fontsize="large")
    axPredicted.imshow(
        correctactual,
        interpolation="nearest",
        cmap="RdYlGn",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    for i in range(len(class_names)):
        txt = axPredicted.text(
            i,
            0,
            (
                "100"
                if correctactual[0, i] == 1.0
                else f"{100.0 * correctactual[0, i]:3.1f}"
            ),
            ha="center",
            va="center",
            color="black",
            fontsize="medium",
        )
        txt.set_path_effects([patheffects.withStroke(linewidth=5, foreground="w")])

    # configure and draw precision percentages
    axActual.set_xticks([0])
    axActual.set_yticks(ticks)
    axActual.set_ylabel("Precision", fontsize="large")
    axActual.yaxis.set_label_position("right")
    axActual.imshow(
        correctpredicted,
        interpolation="nearest",
        cmap="RdYlGn",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    for i in range(len(class_names)):
        txt = axActual.text(
            0,
            i,
            (
                "100"
                if correctpredicted[i, 0] == 1.0
                else f"{100.0 * correctpredicted[i, 0]:3.1f}"
            ),
            ha="center",
            va="center",
            color="black",
            fontsize="medium",
        )
        txt.set_path_effects([patheffects.withStroke(linewidth=5, foreground="w")])

    # configure and draw total accuracy
    axTotal.set_xticks([0])
    axTotal.set_yticks([0])
    axTotal.set_xlabel("Accuracy", fontsize="large")
    axTotal.imshow(
        np.array([[accuracy]]),
        interpolation="nearest",
        cmap="RdYlGn",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    txt = axTotal.text(
        0,
        0,
        f"{100 * accuracy:3.1f}",
        ha="center",
        va="center",
        color="black",
        fontsize="medium",
    )
    txt.set_path_effects([patheffects.withStroke(linewidth=5, foreground="w")])

    # disable axes for other matrices than confusion matrix
    for a in (axPredicted, axActual, axTotal):
        plt.setp(a.get_yticklabels(), visible=False)
        plt.setp(a.get_xticklabels(), visible=False)

    # draw colorbar for confusion matrix
    cbar = fig.colorbar(
        img, ax=[axPredicted, axConfMatrix, axActual, axTotal], shrink=0.5, pad=0.1
    )
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize("medium")
    suptitlehandle = fig.suptitle(f"{title} (ACC={accuracy:.5f})", fontsize="xx-large")
    if outpath is None:
        plt.show()
    else:
        plt.savefig(
            outpath,
            dpi=dpi,
            bbox_inches="tight",
            bbox_extra_artists=[suptitlehandle],
            pad_inches=0.1,
        )
