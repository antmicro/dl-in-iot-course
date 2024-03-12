import numpy as np


def accuracy(confusion_matrix: np.ndarray):
    """
    Computes accuracy of the classifier based on confusion matrix.

    Parameters
    ----------
    confusion_matrix: ArrayLike
        The Numpy nxn array or nxn list representing confusion matrix

    Returns
    -------
    float : accuracy value
    """
    return np.trace(confusion_matrix) / np.sum(confusion_matrix)


def mean_precision(confusion_matrix: np.ndarray):
    """
    Computes mean precision for all classes in the confusion matrix.

    Parameters
    ----------
    confusion_matrix: ArrayLike
        The Numpy nxn array or nxn list representing confusion matrix

    Returns
    -------
    float : mean precision value
    """
    return np.mean(confusion_matrix.diagonal() / np.sum(confusion_matrix, axis=1))


def mean_sensitivity(confusion_matrix: np.ndarray):
    """
    Computes mean sensitivity for all classes in the confusion matrix.

    Parameters
    ----------
    confusion_matrix: ArrayLike
        The Numpy nxn array or nxn list representing confusion matrix

    Returns
    -------
    float : Mean sensitivity
    """
    return np.mean(confusion_matrix.diagonal() / np.sum(confusion_matrix, axis=0))


def g_mean(confusion_matrix: np.ndarray):
    """
    Computes g-mean metric for the confusion matrix.

    Parameters
    ----------
    confusion_matrix: ArrayLike
        The Numpy nxn array or nxn list representing confusion matrix

    Returns
    -------
    float : G-Mean value
    """
    return np.float_power(
        np.prod(
            np.array(confusion_matrix).diagonal() / np.sum(confusion_matrix, axis=0)
        ),
        1.0 / np.array(confusion_matrix).shape[0],
    )
