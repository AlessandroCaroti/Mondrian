from Utility.evaluation import *
from datetime import datetime
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

DATA = None  # Class containing data to anonymize and global ranges and medians
K = 1 # parameter K
NOT_USED_COLUMNS = None # List of columns not used to split yet (to guarantee that each dim is used to split once)
#partition_size = {i: 0 for i in range(1, 100)}

def update_stats(partitions):

    for p in partitions:
        for dim in p.data.columns:
            p.update_width(dim)
            p.update_median(dim)

    return partitions

def compute_normalized_width(partition, dim, norm_factor):
    width = partition.width[dim]

    return width / norm_factor  # normalized with statistic of the original dimension

def chose_dimension(partition, columns, use_all_dim = False):
    """
    :param columns: list of columns
    :param partition: partition to split
    :return: the dimension with max width and which allow cut, and the partitions list
    """
    global DATA, NOT_USED_COLUMNS

    # remove not necessary dimensions
    filtered_dim = filter(lambda item: item[0] in columns, DATA.width_list.items())

    # compute normalized width
    width_map = map(lambda item: [item[0], compute_normalized_width(partition, item[0], item[1])], filtered_dim)

    width_list = list(width_map)  # convert to list

    if len(width_list) == 0:  # no columns allow cut
        return None

    width_list.sort(key=lambda x: x[1], reverse=True)

    i = 0
    # find columns not used yet with maximum width
    if use_all_dim and NOT_USED_COLUMNS:
        while i < len(width_list) and width_list[i][0] in NOT_USED_COLUMNS:
            i += 1

        # all the columns are not used yet
        if i >= len(width_list):
            i = 0

        #filtered_width_list = list(filter(lambda dim: dim[0] in NOT_USED_COLUMNS, width_list))

    return width_list[i][0]  # name of the column


def merge_dictionary(dict_list):
    merged_dict = {}
    for d in dict_list:
        merged_dict = {**merged_dict, **d}

    return merged_dict


def compute_phi(partition):

    global partition_size, num_partition

    #partition_size[len(partition.data.index)] += 1

    summary = []
    for dim in partition.data.columns:

        col_summary = partition.compute_phi(dim)
        summary.append(col_summary)

    # assign the summary created to each row present in the current partition
    phi = {idx: summary for idx in partition.data.index}
    return phi

def allowable_cut(partition_list):
    global K

    if len(partition_list) <= 1:
        return False

    return np.all([len(p.data.index) >= K for p in partition_list])  # strict and relaxed version


def anonymize(partition, use_all_dim = False, update = True):

    global NOT_USED_COLUMNS

    columns = partition.data.columns.tolist()

    while columns:

        dim = chose_dimension(partition, columns, use_all_dim)
        split_val = partition.find_median(dim)
        partition_list = partition.split_partition(dim, split_val)

        # If not allowed multidimensional cut for partition
        if not allowable_cut(partition_list):
            print(dim)
            columns.remove(dim)
            continue

        # the median and width can change after cut so recompute it...
        if update:
            partition_list = update_stats(partition_list)

        # dim is used to split so remove it
        if use_all_dim:
            NOT_USED_COLUMNS.remove(dim)

        print(dim)
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

    global DATA, K, NOT_USED_COLUMNS

    DATA = data
    K = args.K

    if args.use_all_col:
        # copy columns to anonymize
        NOT_USED_COLUMNS = DATA.partition_to_anonymize.data.columns.tolist()

    print("START MONDRIAN")
    if args.show_statistics or args.save_statistics:
        t0 = datetime.now()

    # ANONYMIZE QUASI-IDENTIFIERS: find phi function
    dict_phi = anonymize(data.partition_to_anonymize, args.use_all_col, True)

    if args.show_statistics or args.save_statistics:
        t1 = datetime.now()

    # anonymize the partition with all the dataset
    df_anonymize = anonymization(data.partition_to_anonymize.data, dict_phi)

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
