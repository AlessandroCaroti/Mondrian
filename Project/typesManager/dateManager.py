from datetime import datetime

import pandas as pd
import numpy as np

from dataset_generator.database_generator import random_Bday
from typesManager.abstractType import AbstractType
import Utility.partition as pa


class DateManager(AbstractType):
    data_format = "%d-%m-%Y"
    _max = datetime.strptime("01/01/1000", "%d/%m/%Y")
    _min = datetime.strptime("30/12/3000", "%d/%m/%Y")

    @staticmethod
    def max_min(partition, dim):

        data = partition.data[dim]

        _max = DateManager._max
        _min = DateManager._min

        obj_list = [datetime.strptime(d, DateManager.data_format) for d in data]
        for data_obj in obj_list:
            if data_obj > _max:
                _max = data_obj
            if data_obj < _min:
                _min = data_obj

        return _max, _min

    @staticmethod
    def width(partition, dim):

        data = partition.data[dim]

        if len(data) == 1 or len(data) == 0:
            return 0

        _max, _min = DateManager.max_min(partition, dim)

        return int((_max - _min).total_seconds())

    @staticmethod
    def split(partition_to_split, dim, split_val: str):

        data = partition_to_split.data

        left_idx, right_idx, center_idx = [], [], []
        split_val = datetime.strptime(split_val, DateManager.data_format)
        obj_list = [datetime.strptime(d, DateManager.data_format) for d in
                    data[dim]]  # convert every row of the DataFrame

        # iterate on dim values
        for idx, date in enumerate(obj_list):
            if date > split_val:
                left_idx.append(idx)
            elif date < split_val:
                right_idx.append(idx)
            else:
                center_idx.append(idx)

        mid = len(center_idx) // 2

        left, right, center = data.iloc[left_idx], data.iloc[right_idx], data.iloc[center_idx]

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
        left_width[dim] = DateManager.width(left_p, dim)
        left_median[dim] = DateManager.median(left_p, dim)
        # assign to partition
        left_p.width = left_width
        left_p.median = left_median

        right_width = partition_to_split.width.copy()
        right_median = partition_to_split.median.copy()
        # update width and median
        right_width[dim] = DateManager.width(right_p, dim)
        right_median[dim] = DateManager.median(right_p, dim)
        # assign to partition
        right_p.width = right_width
        right_p.median = right_median

        return [left_p, right_p]

    @staticmethod
    def median(partition, dim):

        data = partition.data[dim]

        obj_array = np.array([datetime.strptime(d, DateManager.data_format) for d in data])

        val_list, frequency = np.unique(obj_array, return_counts=True)
        middle = len(data) // 2

        if len(val_list) == 0:
            return None

        acc = 0
        split_index = 0
        for idx, val in enumerate(val_list):
            acc += frequency[idx]
            if acc >= middle:
                split_index = idx
                break

        split_val = val_list[split_index].strftime(DateManager.data_format)
        return split_val

    @staticmethod
    def summary_statistic(partition, dim) -> str:

        data = partition.data[dim]

        if len(np.unique(data)) == 1:
            return data.iloc[0]

        _max, _min = DateManager.max_min(partition, dim)

        return "[" + \
               _min.strftime(DateManager.data_format) + \
               " - " + \
               _max.strftime(DateManager.data_format) + "]"


def test():
    n_sample = 20
    np.random.seed(42)
    ages = np.random.randint(0, 120, (n_sample,))
    b_day = pd.DataFrame([random_Bday(age) for age in ages])
    print(b_day)

    bday_p = pa.Partition(b_day, {}, {})
    median = DateManager.median(bday_p, 0)
    l, r = DateManager.split(bday_p, 0, median)

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

    print("DIFFERENCE:", DateManager.width(bday_p, 0))
    pass


if __name__ == "__main__":
    test()
