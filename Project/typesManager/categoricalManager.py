from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from DGH.dgh import CsvDGH
import Utility.partition as pa
from typesManager.abstractType import AbstractType


class CategoricalManager(AbstractType):


    @staticmethod
    def width(partition, dim):
        # width as number of distinct values in the partition
        return len(np.unique(partition.data[dim]))

    @staticmethod
    def split(partition_to_split, dim, median_node):

        data = partition_to_split.data
        median_list = partition_to_split.median
        width_list = partition_to_split.width

        new_partition_list = []  # list of the new partitions

        for value, child in median_node.children.items():
            new_median_list = median_list.copy()
            new_width_list = width_list.copy()

            list_idx = []  # list of indexes

            for i, d in enumerate(data[dim]):
                # if the element is in the set, then the tuple is added to the partition
                if str(d) in child.leaf:
                    list_idx.append(i)

            # new median as the child representing the new partition
            new_median_list[dim] = child
            new_width_list[dim] = len(child.leaf)

            # create the new partition
            p = pa.Partition(data.iloc[list_idx], partition_to_split.col_type, new_width_list, new_median_list)

            new_partition_list.append(p)

        # if there is no child the partition cannot be divided and empty list is returned
        return new_partition_list

    @staticmethod
    def median(partition, dim):
        # update the median for the dim as the minimal Node representing the partition
        unique = np.unique(partition.data[dim])
        node = partition.median[dim]
        minimal = node.find_minimal_node(unique)

        return minimal

    @staticmethod
    def summary_statistic(partition, dim):
        return partition.median[dim].data




def prova():
    dgh = CsvDGH("test.csv")
    root = dgh.hierarchy.root

    dgh.hierarchy.print_leaf()

    n_sample = 20
    np.random.seed(42)
    ages = np.random.randint(0, 8, (n_sample,))
    ages = pd.DataFrame(ages)

    bday_p = pa.Partition(ages, {0: len(root.leaf)}, {0: root})
    median = CategoricalManager.median(bday_p, 0)
    [l, r] = CategoricalManager.split(bday_p, 0, median)  # I know there are two partitions because of the
    # Hierarchies
    # (it's a particular case)

    print("MEDIAN:", median.data)
    print("RANGE:", CategoricalManager.summary_statistic(bday_p, 0))

    print("LEFT_PART:")
    print(l.data)
    print("Width :", l.width)
    print("Median :", l.median)

    print()
    print("RIGHT_PART:")
    print(r.data)
    print("Width :", r.width)
    print("Median :", r.median)

    print()

    print("WIDTH:", CategoricalManager.width(bday_p, 0))
    pass


if __name__ == "__main__":
    prova()
