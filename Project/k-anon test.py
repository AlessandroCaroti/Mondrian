from numbers import Number

import numpy as np
from pandas.api.types import is_numeric_dtype


def chose_dimension(dimensions, step):
    dim_pos = int(step % len(dimensions))
    return dimensions[dim_pos]


def merge_dictionary(dict1, dict2):
    return dict2.update(dict2)


def compute_phi(partition):
    np_partition = partition.to_numpy()
    summary = np.empty((np_partition.shape[0],), str)

    for col in range(np_partition.shape[0]):
        _max = np.max(np_partition[col, :])
        _min = np.min(np_partition[col, :])
        summary[col] = "[" + str(_min) + "-" + str(_max) + "]"

    phi = {}
    for i in range(len(partition.index)):
        phi[partition.iloc(i)] = summary
    return phi


def find_median(partition, dim):
    if is_numeric_dtype(partition[dim]):
        return partition[dim].median()

    # todo: gestire le colonne che non sono numeri
    return None


def split_partition(partition, dim, split_val):
    if isinstance(split_val, Number):
        left_p = partition[partition[dim] >= split_val]
        right_p = partition[partition[dim] < split_val]
    else:  # TODO: da gestire il caso in cui non sia un numero
        left_p, right_p = None, None

    return left_p, right_p


def allowable_cut(partition, dim, split_val, k):
    # TODO: test if is allowable multidimensional cut for partition
    value_list = partition[dim].unique()
    if len(value_list) <= 1:
        return False
    if np.where(value_list == split_val) < k:
        return False
    return True


def anonymize(partition, columns, step, k):
    dim = chose_dimension(columns, step)
    mean = find_median(partition, dim)

    # If not allowed multidimensional cut for partition
    if not allowable_cut(partition, dim, mean, k):
        return compute_phi(partition)  # return phi: partition -> summary

    lhs, rhs = split_partition(partition, dim, mean)

    return merge_dictionary(anonymize(lhs, columns, step + 1, k),
                            anonymize(rhs, columns, step + 1, k))
