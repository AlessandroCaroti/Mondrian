import collections
from numbers import Number

import numpy as np
import pandas as pd
from numpy import random
from pandas.api.types import is_numeric_dtype

from typesManager.dateManager import DataManager

root = "" # folder of the data

initial_ranges = dict() # dictionary with the range of each dim of the original table
dgh_dict = dict() # dictionary of dgh for each categorical dim

dim_type = {"B-day": "date"}

def compute_width(values, dim): # dim dovrebbe servire per le colonne categoriche
    width = 0
    if is_numeric_dtype(values): # range width = max - min
        max_r = max(values)
        min_r = min(values)
        width = max_r - min_r

    else: # TODO: da gestire se non numerico, se categorica dipende dalle foglie della gerarchia
        width = None

    return width

def compute_normalized_width(values, dim):
    width = compute_width(values, dim)
    return width / initial_ranges[dim] # normalized with statistic of the original dimension

def chose_dimension(dimensions, partition, k):
    '''
    :param dimensions: list of columns
    :param partition: partition to split
    :return: the dimension with max width and which allow cut and the median
    '''

    width_map = map(lambda dim: [dim, compute_normalized_width(partition[dim], dim), find_median(partition, dim,k)],
                    dimensions)  # get list of all width and median

    width_filtered = filter(lambda tuple: allowable_cut(partition, tuple[0], tuple[2], k),
                            width_map)  # filter out the dimensions which don't allow cut

    width_list = list(width_filtered)  # convert to list

    if len(width_list) == 0:  # no columns allow cut
        return None, None

    get_width = lambda x: x[1]  # function return the width from the tuple
    width_list.sort(key=get_width, reverse=True)  # sort wrt width, maximum first

    return width_list[0][0], width_list[0][2] # name of the column with max width and median

def merge_dictionary(dict1, dict2):
    return {**dict1, **dict2}

def compute_phi(partition):
    np_partition = partition.to_numpy()
    summary = []

    for col, dim in enumerate(partition.columns):
        if is_numeric_dtype(partition[dim]):
            _max = np.max(np_partition[:, col])
            _min = np.min(np_partition[:, col])
            col_summary = "[" + str(_min) + " - " + str(_max) + "]"
            if _min == _max:
                col_summary = str(_min)
        elif dim in dim_type and dim_type[dim] == 'date':
            date_list = partition[dim].tolist()
            col_summary = DataManager.summary_statistic(date_list)
        else:
            col_summary = "ERROR"
        summary.append(col_summary)

    phi = {}
    for idx in partition.index:
        phi[idx] = summary
        # phi[partition.iloc(idx)] = summary
    return phi


def find_median(partition, dim, k):
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

    if dim in dim_type and dim_type[dim] == 'date':
        date_list = partition[dim].tolist()
        return DataManager.median(date_list, k)
    # todo: gestire le colonne che non sono numeri

    return None

def split_partition(partition, dim, split_val):
    if isinstance(split_val, Number):

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

    elif dim in dim_type and dim_type[dim] == 'date':
        date_list = partition[dim].tolist()

        left_idxs, right_idxs = DataManager.split(date_list, split_val)
        left_p = partition.iloc(left_idxs)
        right_p = partition.iloc(right_idxs)

    else:  # TODO: da gestire il caso in cui non sia un numero
        left_p, right_p = None, None

    return left_p, right_p


def allowable_cut(partition, dim, split_val, k):

    partitions_list = split_partition(partition, dim, split_val)

    return np.all([ len(partition) >= k for partition in partitions_list]) # strict version, NON CAPISCO COME SIA RELAXED


def anonymize(partition, columns, step, k):
    dim, median = chose_dimension(columns, partition, k)

    # If not allowed multidimensional cut for partition
    if dim == None:
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

    # Create dictionary with Range statistic for each QI
    global initial_ranges
    initial_ranges = {col: compute_width(df[col], col) for col in cols_to_anonymize}

    # Load dgh for each categorical dim
    global dgh_dict
    dgh_dict = {col: compute_width(df[col], col) if
                        for col in cols_to_anonymize}

    # ANONYMIZE SEMI-IDENTIFIERS DATA
    dict_phi = anonymize(df, cols_to_anonymize, step=0, k=3)

    df_anonymize = anonymization(df, cols_to_anonymize, dict_phi)
    print(df_anonymize)


if __name__ == "__main__":
    debug()
