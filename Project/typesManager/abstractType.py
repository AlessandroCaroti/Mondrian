from abc import ABC, abstractmethod


class AbstractType(ABC):

    @staticmethod
    @abstractmethod
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
        pass

    @staticmethod
    @abstractmethod
    def median(el_list, k: int):
        """
        Compute the median along the input given.

        :param k:
        :param el_list: Input list
        :rtype: the median of the input
        """
        pass

    @staticmethod
    @abstractmethod
    def summary_statistic(el_list):
        """
        Return summary statistic along the input given.

        :param el_list: Input list
        :rtype: a string representing a summary statistic of the input list
        """
        pass
