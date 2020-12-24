from abc import ABC, abstractmethod


class AbstractType(ABC):

    @staticmethod
    @abstractmethod
    def compute_width(partition, dim):
        '''
        :param partition: partition to compute the width
        :return: width wrt the
        '''

        pass

    @staticmethod
    @abstractmethod
    def split(partition_to_split, dim, split_val):
        """
        Given an element and a list of element of the same type, split the list in 2 part
        -left_part = np.where(list_to_split <= split_val)
        -right_part = np.where(list_to_split > split_val)

        :param strict: True -> strict partitioning, False -> relax partitioning
        :param partition_to_split: partition to split
        :param dim:
        :param split_val: value used to divide the list given
        :rtype: (left_part, right_part) 2 list with the position
        """
        pass

    @staticmethod
    @abstractmethod
    def median(partition, dim):
        """
        Compute the median along the input given.

        :param k:
        :param partition: Input partition
        :rtype: the median of the input wrt the dim
        """
        pass

    @staticmethod
    @abstractmethod
    def summary_statistic(partition, dim):

        """
        Return summary statistic along the input given.

        :param el_list: Input list
        :rtype: a string representing a summary statistic of the input list
        """
        pass
