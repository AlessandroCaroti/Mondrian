
import Utility.partition as pa
from typesManager.abstractType import AbstractType

import numpy as np
import pandas as pd


class NumericManager(AbstractType):

    @staticmethod
    def width(partition, dim):

        data = partition.data[dim]

        if len(data) == 0:
            return 0

        max_r = max(data)
        min_r = min(data)

        return max_r - min_r

    @staticmethod
    def split(partition_to_split, dim, split_val):

        data = partition_to_split.data

        left = data[data[dim] < split_val]

        right = data[data[dim] > split_val]
        center = data[data[dim] == split_val]

        mid = len(center.index) // 2

        # balanced partitions
        if len(center[: mid + 1].index) > 0:
            left = pd.concat([left, center[:mid + 1]])

        if len(center[mid + 1:].index) > 0:
            right = pd.concat([right, center[mid + 1:]])

        # create the new partition
        left_p = pa.Partition(left, partition_to_split.col_type)
        right_p = pa.Partition(right, partition_to_split.col_type)

        left_width = partition_to_split.width.copy()
        left_median = partition_to_split.median.copy()
        # update width and median
        left_width[dim] = NumericManager.width(left_p, dim)
        left_median[dim] = NumericManager.median(left_p, dim)
        # assign to partition
        left_p.width = left_width
        left_p.median = left_median

        right_width = partition_to_split.width.copy()
        right_median = partition_to_split.median.copy()
        # update width and median
        right_width[dim] = NumericManager.width(right_p, dim)
        right_median[dim] = NumericManager.median(right_p, dim)
        # assign to partition
        right_p.width = right_width
        right_p.median = right_median

        return [left_p, right_p]

    @staticmethod
    def median(partition, dim):

        data = partition.data

        if len(data.index) == 0:
            return 0

        val_list, frequency = np.unique(data[dim], return_counts=True)
        middle = len(data) // 2

        acc = 0
        split_index = 0
        for idx, val in enumerate(val_list):
            acc += frequency[idx]
            if acc >= middle:
                split_index = idx
                break

        return val_list[split_index]

    @staticmethod
    def summary_statistic(partition, dim):

        data = partition.data[dim]

        if len(np.unique(data)) == 1:
            return data.iloc[0]

        _max, _min = np.max(data), np.min(data)

        return "[" + str(_min) + " - " + str(_max) + "]"


def test():
    n_sample = 20
    np.random.seed(42)
    ages = np.random.randint(0, 120, (n_sample,))
    ages = pd.DataFrame(ages)
    print(ages)

    bday_p = pa.Partition(ages, {}, {})
    median = NumericManager.median(bday_p, 0)
    l, r = NumericManager.split(bday_p, 0, median)

    print("MEDIAN:", median)
    print("RANGE:", NumericManager.summary_statistic(bday_p, 0))

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

    print("DIFFERENCE:", NumericManager.width(bday_p, 0))
    pass


if __name__ == "__main__":
    test()
