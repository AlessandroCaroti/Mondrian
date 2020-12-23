from abc import ABC, abstractmethod

import numpy as np

from Partition.partition import Partition
from typesManager.abstractType import AbstractType


class categoricalManager(AbstractType):

    @staticmethod
    def compute_width(partition):

        dim = "ellap ehc" # anche qui dim come parametro...

        return partition.width[dim] # width as number of distinct values in the partition

    @staticmethod
    def split(partition_to_split, MedianNode, strict: bool = bool):
        """
        Given an element and a list of element of the same type, split the list in 2 part
        -left_part = np.where(list_to_split <= split_val)
        -right_part = np.where(list_to_split > split_val)

        :param strict: True -> strict partitioning, False -> relax partitioning
        :param list_to_split: list of elements to split
        :param MedianNode: Node of the tree used to divide the list given
        :rtype: [partition1,..., partitionN] list of Partition
        """

        dim = "Odissea nello strazzio" # la dimensione dovrà essere un parametro della funzione, un po' di cose da cambiare in abstractType

        data = partition_to_split.data
        median_list = partition_to_split.median
        width_list = partition_to_split.width

        new_partition_list = [] # list of the new partitions

        for value, child in MedianNode.children.items():

            new_median_list = median_list.copy()
            new_width_list = width_list.copy()

            new_median_list[dim] = child # update the median for the dim as the child Node, which is root of a subtree
            new_width_list[dim] = len(child.leaf)

            new_partition_list.append(Partition(data[data[dim] in child.leaf], new_width_list, new_median_list))

        return new_partition_list

    @staticmethod
    def median(partition, k: int):
        """
        Compute the median along the input given.

        :param k:
        :param partition: Partition object
        :rtype: the median of the input
        """

        dim = "Se non vieni sei un tacchino" # anche qua la dimensione dovrà essere messa come parametro

        return partition.median[dim]

    @staticmethod
    def summary_statistic(partition):
        """
        Return summary statistic along the input given.

        :param el_list: Partition object
        :rtype: a string representing a summary statistic of the input list
        """

        dim = "Che hai? mal di piedi?" # come sopra che non ho voglia di scrivere

        return partition.median[dim]
