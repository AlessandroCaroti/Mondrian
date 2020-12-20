import collections
from numbers import Number

import numpy as np
import pandas as pd
from numpy import random
from pandas.api.types import is_numeric_dtype

from types.dateType import MyData

dim_type = {"B-day": "date"}


def chose_dimension(dimensions, step):
    dim_pos = int(step % len(dimensions))
    return dimensions[dim_pos]


def merge_dictionary(dict1, dict2):
    return {**dict1, **dict2}


def compute_phi(partition):
    np_partition = partition.to_numpy()
    summary = []

    for col in range(np_partition.shape[1]):
        _max = np.max(np_partition[:, col])
        _min = np.min(np_partition[:, col])
        col_summary = "[" + str(_min) + " - " + str(_max) + "]"
        if _min == _max:
            col_summary = str(_min)
        summary.append(col_summary)

    phi = {}
    for idx in partition.index:
        phi[idx] = summary
        # phi[partition.iloc(idx)] = summary
    return phi


def find_median(partition, dim):
    if is_numeric_dtype(partition[dim]):
        freq = partition[dim].value_counts(sort=True, ascending=True)
        freq_dict = freq.to_dict()
        values_list = freq.index.to_list()
        # TODO: mettere controllo "stop to split the partition"
        #   valutare se mettere il parametro K globale
        middle = len(partition) // 2
        acc = 0
        split_index = 0
        for i, qi_val in enumerate(values_list):
            acc += freq_dict[qi_val]
            if acc >= middle:
                split_index = i
                break
        median = values_list[split_index]

        return median
    if dim_type[dim] == 'date':
        dim_list = partition[dim].tolist()
        return MyData.median(dim_list, 1)
    # todo: gestire le colonne che non sono numeri
    return None


def split_partition(partition, dim, split_val):
    if isinstance(split_val, Number):
        # print("Split_val: ", split_val)
        left_p = partition[partition[dim] > split_val]
        right_p = partition[partition[dim] < split_val]
        # the tuples with split_val are evenly distributed between the two partitions ( RELAXED version ),
        # also the STRICT version is handled
        center = partition[partition[dim] == split_val]

        mid = int(len(center.index) / 2)

        if len(center[:mid + 1].index) > 0:
            left_p = pd.concat([left_p, center[:mid + 1]])
        if len(center[mid + 1:].index) > 0:
            right_p = pd.concat([right_p, center[mid + 1:]])

    else:  # TODO: da gestire il caso in cui non sia un numero
        left_p, right_p = None, None
    # print("LEFT:\n", left_p,"\n\nRIGHT:\n",right_p)
    return left_p, right_p


def allowable_cut(partition, dim, split_val, k):
    lhs, rhs = split_partition(partition, dim, split_val)
    return len(lhs) >= k and len(rhs) >= k  # strict version, NON CAPISCO ME SIA RELAXED


def anonymize(partition, columns, step, k):
    dim = chose_dimension(columns, step)
    median = find_median(partition, dim)
    # If not allowed multidimensional cut for partition
    if not allowable_cut(partition, dim, median, k):
        return compute_phi(partition)  # return phi: partition -> summary

    lhs, rhs = split_partition(partition, dim, median)

    phi_merger = merge_dictionary(anonymize(lhs, columns, step + 1, k),
                                  anonymize(rhs, columns, step + 1, k))

    return phi_merger


def anonymization(df, columns_to_anonymize, anon_dict):
    # Reorder the semi-identifiers anonymize
    dict_phi = collections.OrderedDict(sorted(anon_dict.items()))

    # Crete a Dataframe from the dictionary
    cols_anonymize = [col + "_anon" for col in columns_to_anonymize]
    anon_df = pd.DataFrame.from_dict(dict_phi, orient='index', columns=cols_anonymize)

    # Concatenate the 2 DF
    df_merged = pd.concat([df, anon_df], axis=1, sort=False)
    print(df_merged)

    # Drop anonymize column
    final_db = df_merged.drop(columns_to_anonymize, axis=1)
    return final_db


def debug():
    # GENERATE A TOY DATASET
    n_sample = 30
    n_cols = 2
    cols_to_anonymize = []

    for i in range(n_cols):
        cols_to_anonymize.append("dim" + str(i))

    # Create a toy dataset
    data = random.randint(0, 10, (n_sample, n_cols))
    df = pd.DataFrame(data, columns=cols_to_anonymize)

    # ANONYMIZE SEMI-IDENTIFIERS DATA
    dict_phi = anonymize(df, cols_to_anonymize, step=0, k=3)

    df_anonymize = anonymization(df, cols_to_anonymize, dict_phi)
    print(df_anonymize)


if __name__ == "__main__":
    debug()
