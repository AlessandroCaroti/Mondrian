from abc import ABC, abstractmethod

import numpy as np

from typesManager.abstractType import AbstractType

class categoricalManager(AbstractType):

    @staticmethod
    def split(list_to_split, split_val, strict: bool = bool):
        """
        Given an element and a list of element of the same type, split the list in 2 part
        -left_part = np.where(list_to_split <= split_val)
        -right_part = np.where(list_to_split > split_val)

        :param strict: True -> strict partitioning, False -> relax partitioning
        :param list_to_split: list of elements to split
        :param split_val: value (Node of the tree) used to divide the list given
        :rtype: [partition1,..., partitionN] list of partition
        """
        pass

    @staticmethod
    def median(el_list, k: int):
        """
        Compute the median along the input given.

        :param k:
        :param el_list:
        :rtype: the median of the input
        """
        pass

    @staticmethod
    def summary_statistic(el_list):
        """
        Return summary statistic along the input given.

        :param el_list: Input list
        :rtype: a string representing a summary statistic of the input list
        """
        pass
