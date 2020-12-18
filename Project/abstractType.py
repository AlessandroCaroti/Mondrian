from abc import ABC, abstractmethod


class AbstractType(ABC):

    @abstractmethod
    def __gt__(self, other):
        # grater then
        pass

    @abstractmethod
    def __le__(self, other):
        # less or equal
        pass

    @staticmethod
    @abstractmethod
    def split(list_to_split, split_val):
        """
        Given an element and a list of element of the same type, split the list in 2 part
        -left_part = np.where(list_to_split <= split_val)
        -right_part = np.where(list_to_split > split_val)

        :param list_to_split: list of elements to split
        :param split_val: value used to divide the list given
        :rtype: (left_part, right_part) 2 list with the position
        """
        pass

    @staticmethod
    @abstractmethod
    def median(el_list):
        """
        Compute the median along the input given.

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
