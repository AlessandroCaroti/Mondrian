from datetime import datetime

import numpy as np

from abstractType import AbstractType


class MyData(AbstractType):
    data_format = "%d-%m-%Y"

    @staticmethod
    def split(list_to_split, split_val: str, strict: bool = True):
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
    def median(el_list):
        if not isinstance(el_list, list) or not isinstance(el_list, np.ndarray):
            raise TypeError
        obj_array = np.array([datetime.strptime(el, MyData.data_format) for el in el_list])

        unique, count = np.unique(obj_array, return_counts=True)
        
        pass

    @staticmethod
    def summary_statistic(el_list) -> str:
        _max = datetime.strptime("30/12/3000", "%d-%m-%Y")
        _min = datetime.strptime("01/10/1800", "%d-%m-%Y")

        obj_list = [datetime.strptime(el, MyData.data_format) for el in el_list]

        for data_obj in obj_list:
            if data_obj > _max:
                _max = data_obj
                continue
            if data_obj < _min:
                _min = data_obj

        return "[" + str(_min) + "-" + str(_max) + "]"


def test():
    pass


if __name__ == "__main__":
    test()
