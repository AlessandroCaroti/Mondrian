from datetime import datetime

import numpy as np

from dataset_generator.database_generator import random_Bday
from typesManager.abstractType import AbstractType


class DataManager(AbstractType):
    data_format = "%d-%m-%Y"
    _max = datetime.strptime("01/01/1000", "%d/%m/%Y")
    _min = datetime.strptime("30/12/3000", "%d/%m/%Y")

    @staticmethod
    def max_min(el_list):
        _max = DataManager._max
        _min = DataManager._min

        obj_list = [datetime.strptime(el, DataManager.data_format) for el in el_list]
        for data_obj in obj_list:
            if data_obj > _max:
                _max = data_obj
            if data_obj < _min:
                _min = data_obj

        return _max, _min

    @staticmethod
    def compute_width(el_list):
        if not isinstance(el_list, list) and not isinstance(el_list, np.ndarray):
            raise TypeError("el_list must be a list or a np_array")
        if len(el_list) == 1:
            return 0

        _max, _min = DataManager.max_min(el_list)

        return int((_max - _min).total_seconds())

    @staticmethod
    def split(list_to_split, split_val: str, strict: bool = True):
        if not isinstance(list_to_split, list) and not isinstance(list_to_split, np.ndarray):
            raise TypeError("list_to_split must be a list or a np_array")

        left_idx, right_idx, center_idx = [], [], []
        split_val = datetime.strptime(split_val, DataManager.data_format)
        obj_list = [datetime.strptime(el, DataManager.data_format) for el in list_to_split]

        for idx, date in enumerate(obj_list):
            if date > split_val:
                left_idx.append(idx)
            elif date < split_val:
                right_idx.append(idx)
            else:
                center_idx.append(idx)

        return left_idx, right_idx, center_idx

    @staticmethod
    def median(el_list, k: int):
        if not isinstance(el_list, list) and not isinstance(el_list, np.ndarray):
            raise TypeError("el_list must be a list or a np_array")

        obj_array = np.array([datetime.strptime(el, DataManager.data_format) for el in el_list])

        val_list, frequency = np.unique(obj_array, return_counts=True)
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

        split_val = val_list[split_index].strftime(DataManager.data_format)
        return split_val

    @staticmethod
    def summary_statistic(el_list) -> str:
        if not isinstance(el_list, list) and not isinstance(el_list, np.ndarray):
            raise TypeError("el_list must be a list or a np_array")
        if len(el_list) == 1:
            return el_list[0]

        _max, _min = DataManager.max_min(el_list)

        return "[" + \
               _min.strftime(DataManager.data_format) + \
               " - " + \
               _max.strftime(DataManager.data_format) + "]"


def test():
    n_sample = 20
    np.random.seed(42)
    ages = np.random.randint(0, 120, (n_sample,))
    b_day = [random_Bday(age) for age in ages]
    print(b_day)

    median = DataManager.median(b_day, 1)
    l, r, c = DataManager.split(b_day, median)

    print("MEDIAN:", median)
    print("RANGE:", DataManager.summary_statistic(b_day))

    print("LEFT_PART:")
    for index in l:
        print("", b_day[index], end=",")
    print()
    print("RIGHT_PART:")
    for index in r:
        print("", b_day[index], end=",")
    print()

    print("DIFFERENCE:", DataManager.compute_width(b_day))
    pass


if __name__ == "__main__":
    test()
