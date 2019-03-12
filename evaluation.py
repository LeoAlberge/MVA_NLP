import numpy as np


def accuracy(list_pos_predicted, list_pos_true):
    """

    :param list_pos_predicted:
    :param list_pos_true:
    :return: Accuracy
    """
    return (np.array(list_pos_predicted)==np.array(list_pos_true)).sum()/len(list_pos_predicted)
