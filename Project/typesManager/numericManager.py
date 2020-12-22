from pandas.core.dtypes.common import is_numeric_dtype

from typesManager.abstractType import AbstractType
import numpy as np


class NumericManager(AbstractType):

    @staticmethod
    def compute_width(el_list):
        if not isinstance(el_list, np.ndarray):
            raise TypeError("list_to_split must be a np_array")

        max_r = max(el_list)
        min_r = min(el_list)
        return max_r - min_r

    @staticmethod
    def split(list_to_split, split_val, strict: bool = bool):
        """
        Given an element and a list of element of the same type, split the list in 2 part
        -left_part = np.where(list_to_split <= split_val)
        -right_part = np.where(list_to_split > split_val)

        :param strict: True -> strict partitioning, False -> relax partitioning
        :param list_to_split: list of elements to split
        :param split_val: value used to divide the list given
        :rtype: (left_part, right_part) 2 list with the position
        """
        if not isinstance(list_to_split, np.ndarray):
            raise TypeError("list_to_split must be a np_array")

        left_idx, right_idx, center_idx = [], [], []

        for idx, el in enumerate(list_to_split):
            if el > split_val:
                left_idx.append(idx)
            elif el < split_val:
                right_idx.append(idx)
            else:
                center_idx.append(idx)

        return left_idx, right_idx, center_idx

    @staticmethod
    def median(el_list, k: int):
        """
        Compute the median along the input given.

        :param k:
        :param el_list: Input list
        :rtype: the median of the input
        """
        if not isinstance(el_list, list) and not isinstance(el_list, np.ndarray):
            raise TypeError("list_to_split must be a np_array")

        val_list, frequency = np.unique(el_list, return_counts=True)
        middle = len(el_list) // 2

        # Stop to split the partition todo rimmuovere i commenti
        # if middle < k or len(val_list) <= 1:
        #    return None

        acc = 0
        split_index = 0
        for idx, val in enumerate(val_list):
            acc += frequency[idx]
            if acc >= middle:
                split_index = idx
                break

        return val_list[split_index]

    @staticmethod
    def summary_statistic(el_list):
        """
        Return summary statistic along the input given.

        :param el_list: Input list
        :rtype: a string representing a summary statistic of the input list
        """
        pass
