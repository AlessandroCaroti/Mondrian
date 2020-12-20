from datetime import datetime

import numpy as np

from abstractType import AbstractType


class MyData(AbstractType):
    data_format = "%d-%m-%Y"

    @staticmethod
    def split(list_to_split, split_val: str, strict: bool = True):
        if not isinstance(list_to_split, list) and not isinstance(list_to_split, np.ndarray):
            raise TypeError("list_to_split must be a list or a np_array")

        left_idx, right_idx = [], []
        split_val = datetime.strptime(split_val, MyData.data_format)
        obj_list = [datetime.strptime(el, MyData.data_format) for el in list_to_split]

        for idx, date in enumerate(obj_list):
            if date >= split_val:
                right_idx.append(idx)
            else:
                left_idx.append(idx)

        return left_idx, right_idx

    @staticmethod
    def median(el_list, k: int):
        if not isinstance(el_list, list) and not isinstance(el_list, np.ndarray):
            raise TypeError("el_list must be a list or a np_array")

        obj_array = np.array([datetime.strptime(el, MyData.data_format) for el in el_list])

        val_list, frequency = np.unique(obj_array, return_counts=True)
        middle = len(el_list) // 2

        # Stop to split the partition
        if middle < k or len(val_list) <= 1:
            return None

        acc = 0
        split_index = 0
        for idx, val in enumerate(val_list):
            acc += frequency[idx]
            if acc >= middle:
                split_index = idx
                break

        split_val = val_list[split_index].strftime(MyData.data_format)
        return split_val, True

    @staticmethod
    def summary_statistic(el_list) -> str:
        if not isinstance(el_list, list) and not isinstance(el_list, np.ndarray):
            raise TypeError("el_list must be a list or a np_array")

        _max = datetime.strptime("01/10/1800", "%d/%m/%Y")
        _min = datetime.strptime("30/12/3000", "%d/%m/%Y")

        obj_list = [datetime.strptime(el, MyData.data_format) for el in el_list]

        for data_obj in obj_list:
            if data_obj > _max:
                _max = data_obj
            elif data_obj < _min:
                _min = data_obj

        return "[" + \
               _min.strftime(MyData.data_format) + \
               " - " + \
               _max.strftime(MyData.data_format) + "]"


def random_Bday(age):
    day = np.random.randint(1, 28)
    mouth = np.random.randint(1, 12)
    year = datetime.now().year - age
    return "{:02d}-{:02d}-{}".format(day, mouth, year)


def test():
    n_sample = 20
    ages = np.random.randint(0, 120, (n_sample,))
    b_day = [random_Bday(age) for age in ages]
    print(b_day)

    median = MyData.median(b_day, 1)[0]
    l, r = MyData.split(b_day, median)

    print("MEDIAN:", median)
    print("RANGE:", MyData.summary_statistic(b_day))

    print("LEFT_PART:")
    for index in l:
        print("", b_day[index], end=",")
    print()
    print("RIGHT_PART:")
    for index in r:
        print("", b_day[index], end=",")
    pass


if __name__ == "__main__":
    test()
