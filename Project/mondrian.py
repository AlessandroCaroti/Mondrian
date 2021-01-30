from Utility.evaluation import *
from datetime import datetime
import numpy as np
import pandas as pd

DATA = None  # Class containing data to anonymize and global ranges and medians
K = 1  # parameter K
N_PARTITIONS = 0 # count the number of partitions


def init(data=None, k=1, n_partition=0):
    global DATA, K, N_PARTITIONS
    DATA = data
    K = k
    N_PARTITIONS = n_partition


def update_stats(partitions):
    for p in partitions:
        for dim in p.data.columns:
            p.update_width(dim)
            p.update_median(dim)

    return partitions


def compute_normalized_width(partition, dim, norm_factor):
    width = partition.get_width(dim)

    return width / norm_factor  # normalized with statistic of the original dimension


def chose_dimension(partition, columns, first=False):
    """
    :param first: if first time split
    :param columns: list of columns
    :param partition: partition to split
    :return: the dimension with max width and which allow cut, and the partitions list
    """
    global DATA

    # remove not necessary dimensions
    filtered_dim = filter(lambda item: item[0] in columns, DATA.width_list.items())

    if first:
        # the first time the function is called it makes no sense normalize
        width_map = map(lambda item: [item[0], partition.get_width(item[0])], filtered_dim)
    else:
        # compute normalized width
        width_map = map(lambda item: [item[0], compute_normalized_width(partition, item[0], item[1])], filtered_dim)

    width_list = list(width_map)  # convert to list

    if len(width_list) == 0:  # no columns allow cut
        return None

    width_list.sort(key=lambda x: x[1], reverse=True)

    return width_list[0][0]  # name of the column with max width


def merge_dictionary(dict_list):
    merged_dict = {}
    for d in dict_list:
        merged_dict = {**merged_dict, **d}

    return merged_dict


def compute_phi(partition):
    global N_PARTITIONS

    N_PARTITIONS += 1
    summary = []
    for dim in partition.data.columns:
        col_summary = partition.compute_phi(dim)
        summary.append(col_summary)

    # assign the summary created to each row present in the current partition
    phi = {idx: summary for idx in partition.data.index}
    return phi


def allowable_cut(partition_list):
    global K

    if len(partition_list) < 1:
        return False

    return np.all([len(p.data.index) >= K for p in partition_list])


def anonymize(partition, first=False):

    columns = partition.data.columns.tolist()
    while columns:

        dim = chose_dimension(partition, columns, first)
        split_val = partition.get_median(dim)
        partition_list = partition.split_partition(dim, split_val)

        # If not allowed multidimensional cut for partition
        if not allowable_cut(partition_list):
            columns.remove(dim)
            continue

        # the median and width can change after cut so recompute it...
        partition_list = update_stats(partition_list)

        return merge_dictionary([anonymize(p) for p in partition_list])

    return compute_phi(partition)  # return phi: partition -> summary


def anonymization(df, anon_dict):

    # Reorder the semi-identifiers anonymize
    dict_phi = {k: anon_dict[k] for k in sorted(anon_dict)}
    columns_to_anonymize = list(df.columns)

    # Crete a Dataframe from the dictionary
    cols_anonymize = [col + "_anon" for col in columns_to_anonymize]

    anon_df = pd.DataFrame.from_dict(dict_phi, orient='index', columns=cols_anonymize)

    # Concatenate the 2 DF
    df_merged = pd.concat([df, anon_df], axis=1, sort=False)

    # Drop anonymize columns
    final_db = df_merged.drop(columns_to_anonymize, axis=1)
    return final_db


def main(args, data):
    global DATA, K, N_PARTITIONS

    DATA = data
    K = args.K

    print("START MONDRIAN")

    t0 = datetime.now()

    # ANONYMIZE QUASI-IDENTIFIERS: find phi function
    dict_phi = anonymize(data.partition_to_anonymize, True)

    t1 = datetime.now()

    # anonymize the partition with all the dataset
    df_anonymize = anonymization(data.partition_to_anonymize.data, dict_phi)

    t2 = datetime.now()

    print("END MONDRIAN\n")

    # save result in a file
    data.data_anonymized = df_anonymize
    data.save_anonymized(K)
    print("\nResult saved!")

    print("Total time:      ", t2 - t0)

    if args.save_info:
        columns = df_anonymize.columns.tolist()

        cavg = c_avg(N_PARTITIONS, df_anonymize, K)

        save_info(DATA.get_path_results(), cavg, t0, t1, t2, N_PARTITIONS, len(df_anonymize.index),
                  len(columns), K)
