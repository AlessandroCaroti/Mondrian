from datetime import datetime
from numbers import Number

import numpy as np
import pandas as pd
from numpy import random
from pandas.api.types import is_numeric_dtype

from typesManager.dateManager import DateManager
from dataset_generator.database_generator import random_Bday
from typesManager.numericManager import NumericManager


initial_ranges = {}
dim_type = {"B-day": "date"}
num_partition = 0
partition_size = {i: 0 for i in range(1, 13)}


def compute_width(values, dim):  # dim dovrebbe servire per le colonne categoriche
    if is_numeric_dtype(values):  # range width = max - min
        list_np = values.to_numpy()
        width = NumericManager.compute_width(list_np)

    elif dim in dim_type and dim_type[dim] == 'date':
        date_list = values.tolist()
        width = DateManager.compute_width(date_list)

    else:
        raise Exception("WITH")  # TODO: manage categorical data

    return width


def compute_normalized_width(values, dim):
    width = compute_width(values, dim)

    return width / initial_ranges[dim]  # normalized with statistic of the original dimension


def chose_dimension(dimensions, partition):
    """
    :param dimensions: list of columns
    :param partition: partition to split
    :return: the dimension with max width and which allow cut
    """

    width_map = map(lambda dim: [dim, compute_normalized_width(partition[dim], dim)],
                    dimensions)  # get list of all width and median

    width_list = list(width_map)  # convert to list
    width_list.sort(key=lambda x: x[1], reverse=True)

    return width_list[0][0]  # name of the column with max width


def merge_dictionary(dict1, dict2):
    return {**dict1, **dict2}


def compute_phi(partition):
    summary = []
    global partition_size
    partition_size[len(partition)] += 1
    num_partition += 1

    for dim in partition.columns:
        if is_numeric_dtype(partition[dim]):
            list_np = partition[dim].to_numpy()
            col_summary = NumericManager.summary_statistic(list_np)
        elif dim in dim_type and dim_type[dim] == 'date':
            date_list = partition[dim].tolist()
            col_summary = DateManager.summary_statistic(date_list)
        else:
            raise Exception("MEDIAN")  # TODO: manage categorical data
        summary.append(col_summary)

    # assign the summary created to each row present in the current partition
    phi = {idx: summary for idx in partition.index}
    return phi


def find_median(partition, dim, k):

    if is_numeric_dtype(partition[dim]):
        list_np = partition[dim].to_numpy()
        return NumericManager.median(list_np, k)

    if dim in dim_type and dim_type[dim] == 'date':
        date_list = partition[dim].tolist()
        return DateManager.median(date_list, k)

    raise Exception("MEDIAN")  # TODO: manage categorical data


def split_partition(partition, dim, split_val):
    if isinstance(split_val, Number):
        list_np = partition[dim].to_numpy()

        left_idx, right_idx, center_idx = NumericManager.split(list_np, split_val)

    elif dim in dim_type and dim_type[dim] == 'date':
        date_list = partition[dim].tolist()

        left_idx, right_idx, center_idx = DateManager.split(date_list, split_val)

    else:  # TODO: manage categorical data
        raise Exception("SPLIT_CATEGORICAL")

    mid = len(center_idx) // 2
    left_p, right_p, center = partition.iloc[left_idx], partition.iloc[right_idx], partition.iloc[center_idx]

    if len(center_idx[:mid + 1]) > 0:
        left_p = pd.concat([left_p, center[:mid + 1]])

    if len(center_idx[mid + 1:]) > 0:
        right_p = pd.concat([right_p, center[mid + 1:]])
    return left_p, right_p


def anonymize(partition, k):
    columns = partition.columns.tolist()

    while columns and len(partition) >= k * 2:
        dim = chose_dimension(columns, partition)  # chooses the dimension with the widest normalized range
        median = find_median(partition, dim, k)  # compute the frequency set and find the median
        lhs, rhs = split_partition(partition, dim, median)  #

        # check if, for the current dim, lhs and rhs satisfy k-anonymity
        if len(lhs) < k or len(rhs) < k:
            columns.remove(dim)
            continue

        return merge_dictionary(anonymize(lhs, k),
                                anonymize(rhs, k))

    return compute_phi(partition)  # return phi: partition -> summary


def anonymization(df, columns_to_anonymize, anon_dict):
    # Reorder the semi-identifiers anonymize
    dict_phi = {k: anon_dict[k] for k in sorted(anon_dict)}

    # Crete a Dataframe from the dictionary
    cols_anonymize = [col + "_anon" for col in columns_to_anonymize]
    anon_df = pd.DataFrame.from_dict(dict_phi, orient='index', columns=cols_anonymize)

    # Concatenate the 2 DF
    df_merged = pd.concat([df, anon_df], axis=1, sort=False)

    # Drop anonymize columns
    final_db = df_merged.drop(columns_to_anonymize, axis=1)
    return final_db


def toy_dataset():
    # GENERATE A TOY DATASET
    n_sample = 10000
    n_cols = 3
    col_list = ["dim" + str(i) for i in range(n_cols)]
    all_data = np.empty((n_sample, 0), dtype=np.object)

    # Create a toy dataset
    random.seed(42)
    data = random.randint(0, 50, (n_sample, n_cols)).astype(int)
    all_data = np.append(all_data, data, axis=1)

    # Add date to the data
    b_day = np.array([random_Bday(age) for age in np.random.randint(0, 120, (n_sample,))]).reshape((n_sample, 1))
    # all_data = np.append(all_data, b_day, axis=1)
    # col_list.append("B-day")

    df = pd.DataFrame(all_data, columns=col_list)
    df = df.infer_objects()
    df = df.convert_dtypes()

    return df, col_list


def debug():
    df, cols_to_anonymize = toy_dataset()
    k = 3

    # Create dictionary with Range statistic for each QI
    global initial_ranges
    initial_ranges = {col: compute_width(df[col], col) for col in cols_to_anonymize}

    # ANONYMIZE SEMI-IDENTIFIERS DATA
    t0 = datetime.now()
    dict_phi = anonymize(df, k)
    t1 = datetime.now()

    df_anonymize = anonymization(df, cols_to_anonymize, dict_phi)
    t2 = datetime.now()

    print("n_row:{}  -  n_dim:{}  -  k:{}".format(len(df), len(cols_to_anonymize), k))
    print("-Partition created:", sum(partition_size.values()))
    print("-Total time:      ", t2 - t0)
    print("-Compute phi time:", t1 - t0)
    print(partition_size)
    print("_________________________________________________________")
    print(df_anonymize)


if __name__ == "__main__":
    debug()
