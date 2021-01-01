from datetime import datetime
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from Utility.data import Data
from Utility.evaluation import *
from typesManager.dateManager import DateManager
from typesManager.numericManager import NumericManager
from typesManager.categoricalManager import CategoricalManager

DATA = None  # Class containing data to anonymize and global ranges and medians
K = 1 # parameter K

#partition_size = {i: 0 for i in range(1, 100)}


def compute_normalized_width(partition, dim, norm_factor):
    width = partition.width[dim]

    return width / norm_factor  # normalized with statistic of the original dimension


def chose_dimension(partition, columns):
    """
    :param columns: list of columns
    :param partition: partition to split
    :return: the dimension with max width and which allow cut, and the partitions list
    """
    global DATA

    # remove not necessary dimensions
    filtered_dim = filter(lambda item: item[0] in columns, DATA.width_list.items())
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

    global partition_size, num_partition, DATA

    #partition_size[len(partition.data.index)] += 1

    col_summary = None
    summary = []
    for dim in partition.data.columns:
        if DATA.dim_QI[dim] == DATA.NUMERICAL:
            col_summary = NumericManager.summary_statistic(partition, dim)

        if DATA.dim_QI[dim] == DATA.DATE:
            col_summary = DateManager.summary_statistic(partition, dim)

        if DATA.dim_QI[dim] == DATA.CATEGORICAL:
            col_summary = CategoricalManager.summary_statistic(partition, dim)

        summary.append(col_summary)

    # assign the summary created to each row present in the current partition
    phi = {idx: summary for idx in partition.data.index}
    return phi


def find_median(partition, dim):
    global DATA

    if DATA.dim_QI[dim] == DATA.NUMERICAL:
        return NumericManager.median(partition, dim)

    if DATA.dim_QI[dim] == DATA.DATE:
        return DateManager.median(partition, dim)

    if DATA.dim_QI[dim] == DATA.CATEGORICAL:
        return CategoricalManager.median(partition, dim)


def split_partition(partition, dim, split_val):
    global DATA

    if DATA.dim_QI[dim] == DATA.NUMERICAL:
        left, right = NumericManager.split(partition, dim, split_val)
        return [left, right]

    if DATA.dim_QI[dim] == DATA.DATE:
        left, right = DateManager.split(partition, dim, split_val)
        return [left, right]

    if DATA.dim_QI[dim] == DATA.CATEGORICAL:
        partition_list = CategoricalManager.split(partition, dim, split_val)
        return partition_list


def allowable_cut(partition_list):
    global K

    if len(partition_list) <= 1:
        return False

    return np.all([len(p.data.index) >= K for p in partition_list])  # strict and relaxed version


def anonymize(partition):
    columns = partition.data.columns.tolist()

    while columns:

        dim = chose_dimension(partition, columns)
        split_val = find_median(partition, dim)
        partition_list = split_partition(partition, dim, split_val)

        # If not allowed multidimensional cut for partition
        if not allowable_cut(partition_list):
            columns.remove(dim)
            continue

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

    global DATA, K
    DATA = data
    K = args.K


    print("START MONDRIAN...")
    if args.show_statistics or args.save_statistics:
        t0 = datetime.now()

    # ANONYMIZE QUASI-IDENTIFIERS: find phi function
    dict_phi = anonymize(data.data_to_anonymize)

    if args.show_statistics or args.save_statistics:
        t1 = datetime.now()

    # anonymize the partition with all the dataset
    df_anonymize = anonymization(data.data_to_anonymize.data, dict_phi)

    if args.show_statistics or args.save_statistics:
        t2 = datetime.now()
    print("END MONDRIAN\n")

    # save result in a file
    data.data_anonymized = df_anonymize
    data.save_anonymized()
    print("Result saved!")

    if args.show_statistics:
        print("Total time:      ", t2 - t0)
        #print("-Compute phi time:", t1 - t0)

    '''
        if args.show_statistics or args.save_statistics:
        equivalence_classes = get_equivalence_classes(df_anonymize, df_anonymize.columns)

    if args.show_statistics:
        print("CONDITION: C_dm >= k * total_records: ")
        print(str(c_dm(equivalence_classes)), ">=", str(K), "*", str(len(df_anonymize)), ": "
            , str(c_dm(equivalence_classes) >= (K * len(df_anonymize))))

        print("CONDITION: C_avg >= 1: ")

        print(str(c_avg(equivalence_classes, df_anonymize, K)), ">= 1: ",
            str(c_avg(equivalence_classes, df_anonymize, K) >= 1))
        
    '''
