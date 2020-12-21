from abc import ABC, abstractmethod

import numpy as np
from pandas.core.dtypes.common import is_numeric_dtype

from typesManager.abstractType import AbstractType


class numericManager(AbstractType):

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
    def find_median(partition, dim, k: int):

        """
        Compute the median along the input given.
        :param dim:
        :param partition:
        :param k:
        :rtype: the median of the input
        """
        if is_numeric_dtype(partition[dim]):
            freq = partition[dim].value_counts(sort=True, ascending=True)
            freq_dict = freq.to_dict()
            values_list = freq.index.to_list()
            # TODO: mettere controllo "stop to split the partition"
            #   valutare se mettere il parametro K globale
            middle = len(partition) // 2
            acc = 0
            split_index = 0
            for i, qi_val in enumerate(values_list):
                acc += freq_dict[qi_val]
                if acc >= middle:
                    split_index = i
                    break
            median = values_list[split_index]

            return median

        if dim in dim_type and dim_type[dim] == 'date':
            date_list = partition[dim].tolist()
            return DataManager.median(date_list, k)

        pass

    @staticmethod
    def summary_statistic(el_list):
        """
        Return summary statistic along the input given.

        :param el_list: Input list
        :rtype: a string representing a summary statistic of the input list
        """
        pass
