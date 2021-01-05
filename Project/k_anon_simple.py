import os
from datetime import datetime
from numbers import Number

from pathlib import Path
import numpy as np
import pandas as pd
from numpy import random
from pandas.api.types import is_numeric_dtype

from Mondrian.Utility.data import Data
from Mondrian.typesManager.dateManager import DateManager
from Mondrian.typesManager.numericManager import NumericManager
from Mondrian.typesManager.categoricalManager import CategoricalManager
from Mondrian.Utility.evaluation import *
import matplotlib.pyplot as plt

data = None  # Class containing data to anonymize and global ranges and medians
dim_type = {"B-day": "date"}
partition_size = {i: 0 for i in range(1, 100)}

K = 1


def compute_normalized_width(partition, dim, norm_factor):
    width = partition.width[dim]

    return width / norm_factor  # normalized with statistic of the original dimension


def chose_dimension(partition, columns):
    """
    :param columns: list of columns
    :param partition: partition to split
    :return: the dimension with max width and which allow cut, and the partitions list
    """
    global data

    # remove not necessary dimensions
    filtered_dim = filter(lambda item: item[0] in columns, data.width_list.items())
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
    summary = []
    global partition_size, num_partition, data
    partition_size[len(partition.data.index)] += 1

    col_summary = None
    for dim in partition.data.columns:
        if data.columns_type[dim] == Data.NUMERICAL:
            col_summary = NumericManager.summary_statistic(partition, dim)

        if data.columns_type[dim] == Data.DATE:
            col_summary = DateManager.summary_statistic(partition, dim)

        if data.columns_type[dim] == Data.CATEGORICAL:
            col_summary = CategoricalManager.summary_statistic(partition, dim)

        summary.append(col_summary)

    # assign the summary created to each row present in the current partition
    phi = {idx: summary for idx in partition.data.index}
    return phi


def find_median(partition, dim):
    global data

    if data.columns_type[dim] == Data.NUMERICAL:
        return NumericManager.median(partition, dim)

    if data.columns_type[dim] == Data.DATE:
        return DateManager.median(partition, dim)

    if data.columns_type[dim] == Data.CATEGORICAL:
        return CategoricalManager.median(partition, dim)


def split_partition(partition, dim, split_val):
    global data

    if data.columns_type[dim] == Data.NUMERICAL:
        left, right = NumericManager.split(partition, dim, split_val)
        return [left, right]

    if data.columns_type[dim] == Data.DATE:
        left, right = DateManager.split(partition, dim, split_val)
        return [left, right]

    if data.columns_type[dim] == Data.CATEGORICAL:
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


from Mondrian.dataset_generator.database_generator import random_Bday


def toy_dataset():
    # GENERATE A TOY DATASET
    n_sample = 10
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


def debug_real_dataset():
    global K, data
    K = 5

    dataset_folder = "Dataset_real/adult_final.csv"

    dataset = pd.read_csv(os.path.join(dataset_folder))
    cols = {'age': Data.NUMERICAL, 'workclass': Data.CATEGORICAL, 'education': Data.CATEGORICAL,
            'martial-status': Data.CATEGORICAL, 'occupation': Data.CATEGORICAL, 'relationship': Data.CATEGORICAL,
            'race': Data.CATEGORICAL, 'sex': Data.CATEGORICAL, 'captital-gain': Data.NUMERICAL,
            'capital-loss': Data.NUMERICAL, 'native-country': Data.CATEGORICAL, 'hours-per-week': Data.NUMERICAL,
            'annual-gain': Data.SD}

    data = Data(dataset, cols)
    # ANONYMIZE QUASI-IDENTIFIERS DATA
    t0 = datetime.now()
    dict_phi = anonymize(data.data_to_anonymize)
    t1 = datetime.now()
    df_anonymize = anonymization(data.data_to_anonymize.data, dict_phi)
    t2 = datetime.now()

    data.data_anonymized = df_anonymize
    len_dataset = len(dataset)
    n_dim = len(list(data.data_to_anonymize.data))
    print("n_row:{}  -  n_dim:{}  -  k:{}".format(len_dataset, n_dim, K))
    total_partition = sum(partition_size.values())
    print("-Partition created:", total_partition)
    print("-Total time:      ", t2 - t0)
    print("-Compute phi time:", t1 - t0)
    print(partition_size)
    print("__________________________________________________________")
    print(df_anonymize)

    if not os.path.isdir("results/real"):
        os.makedirs("results/real")

    df_anonymize.to_csv(os.path.join("results", "real", "Anonymized_Dataset_real_K_"+str(K)+".csv "))

    equivalence_classes = get_equivalence_classes(df_anonymize, list(data.data_anonymized))

    equivalence_classes.to_csv(os.path.join("results", "real", "Equivalence_Classes_real_K_"+str(K)+".csv"))
    print("\n\nEquivalence Classes:\n\n", equivalence_classes)

    print("\n\n-------------------------------------EVALUATION---------------------------------------------\n\n")

    print("CONDITION: C_dm >= k * total_records: ")

    cdm = c_dm(equivalence_classes)
    print(str(cdm), ">=", str(K), "*", str(len(df_anonymize)), ": "
          , str(cdm >= (K * len(df_anonymize))))

    print("CONDITION: C_avg >= 1: ")

    cavg = c_avg(equivalence_classes, df_anonymize, K)
    print(str(cavg), ">= 1: ",
          str(cavg >= 1))

    # SAVE ALL STATISTICS IN THE FOLDER RESULTS
    f = open("results/real/statistics_result_real_K_"+str(K)+".txt", "w")
    f.write("\n---------------------------------EVALUATION-STATISTICS-------------------------------------------\n")
    f.write("\nDiscernability Penalty Metric: {}\n".format(cdm))
    f.write("\nNormalized Average Equivalence Class Size Metric: {}\n".format(cavg))
    f.write("\nTotal Execution Time: {}\n".format(t2 - t0))
    f.write("\nExecution Time - Computation PHI: {}\n".format(t1 - t0))
    f.write("\nPartition created: {}\n".format(total_partition))
    f.write("\nSize of the Dataset_synthetic: {}  -  Number of Attribute: {}  -  K: {}".format(len_dataset, n_dim, K))
    f.close()


def debug_dataset():
    global K, data
    K = 3
    dataset_name = "mainDB_100.csv"
    dataset_folder = "dataset_generator/data"
    n_sample_filename = dataset_name.split("_")[1].split(".")[0]

    dataset = pd.read_csv(os.path.join(dataset_folder, dataset_name))
    col_type = {"Name": Data.EI, "Gender": Data.CATEGORICAL, "Age": Data.NUMERICAL, "Zipcode": Data.CATEGORICAL,
                "B-City": Data.CATEGORICAL, 'B-day': Data.DATE, 'Disease': Data.SD, 'Start Therapy': Data.DATE,
                'End Therapy': Data.DATE,
                'Blood type': Data.CATEGORICAL, 'Weight (Kg)': Data.NUMERICAL, 'Height (cm)': Data.NUMERICAL}

    data = Data(dataset, col_type)
    # ANONYMIZE QUASI-IDENTIFIERS DATA
    t0 = datetime.now()
    dict_phi = anonymize(data.data_to_anonymize)
    t1 = datetime.now()
    df_anonymize = anonymization(data.data_to_anonymize.data, dict_phi)
    t2 = datetime.now()

    data.data_anonymized = df_anonymize
    len_dataset = len(dataset)
    n_dim = len(list(data.data_to_anonymize.data))
    print("n_row:{}  -  n_dim:{}  -  k:{}".format(len_dataset, n_dim, K))
    total_partition = sum(partition_size.values())
    print("-Partition created:", total_partition)
    print("-Total time:      ", t2 - t0)
    print("-Compute phi time:", t1 - t0)
    print(partition_size)
    print("__________________________________________________________")
    print(df_anonymize)

    if not os.path.isdir("results"):
        os.mkdir("results")

    df_anonymize.to_csv(os.path.join("results", "Anonymized_Dataset_DB_" + n_sample_filename + ".cvs"))

    equivalence_classes = get_equivalence_classes(df_anonymize, list(list(data.data_anonymized)))

    equivalence_classes.to_csv(os.path.join("results", "Equivalence_Classes_DB_" + n_sample_filename + ".cvs"))
    print("\n\nEquivalence Classes:\n\n", equivalence_classes)

    print("\n\n-------------------------------------EVALUATION---------------------------------------------\n\n")

    print("CONDITION: C_dm >= k * total_records: ")

    cdm = c_dm(equivalence_classes)
    print(str(cdm), ">=", str(K), "*", str(len(df_anonymize)), ": "
          , str(cdm >= (K * len(df_anonymize))))

    print("CONDITION: C_avg >= 1: ")

    cavg = c_avg(equivalence_classes, df_anonymize, K)
    print(str(cavg), ">= 1: ",
          str(cavg >= 1))

    # SAVE ALL STATISTICS IN THE FOLDER RESULTS
    f = open("results/statistics_result_DB_" + n_sample_filename + ".txt", "w")
    f.write("\n---------------------------------EVALUATION-STATISTICS-------------------------------------------\n")
    f.write("\nDiscernability Penalty Metric: {}\n".format(cdm))
    f.write("\nNormalized Average Equivalence Class Size Metric: {}\n".format(cavg))
    f.write("\nTotal Execution Time: {}\n".format(t2 - t0))
    f.write("\nExecution Time - Computation PHI: {}\n".format(t1 - t0))
    f.write("\nPartition created: {}\n".format(total_partition))
    f.write("\nSize of the Dataset_synthetic: {}  -  Number of Attribute: {}  -  K: {}".format(len_dataset, n_dim, K))
    f.close()


def debug():
    df, cols_to_anonymize = toy_dataset()
    global K, data
    K = 3

    print(df)
    # Create dictionary with Range statistic and Median for each QI
    col_type = {"dim0": Data.NUMERICAL, "dim1": Data.NUMERICAL, "dim2": Data.NUMERICAL}
    data = Data(df, col_type)

    # ANONYMIZE QUASI-IDENTIFIERS DATA
    t0 = datetime.now()
    dict_phi = anonymize(data.data_to_anonymize)
    t1 = datetime.now()

    df_anonymize = anonymization(data.data_to_anonymize.data, dict_phi)
    t2 = datetime.now()

    data.data_anonymized = df_anonymize
    print("n_row:{}  -  n_dim:{}  -  k:{}".format(len(df), len(cols_to_anonymize), K))
    print("-Partition created:", sum(partition_size.values()))
    print("-Total time:      ", t2 - t0)
    print("-Compute phi time:", t1 - t0)
    print(partition_size)
    print("__________________________________________________________")
    print(df_anonymize)

    column_list = ['dim0_anon', 'dim1_anon', 'dim2_anon']
    equivalence_classes = get_equivalence_classes(df_anonymize, column_list)

    print("\n\nEquivalence Classes:\n\n", equivalence_classes)

    print("\n\n-------------------------------------EVALUATION---------------------------------------------\n\n")

    print("CONDITION: C_dm >= k * total_records: ")
    print(str(c_dm(equivalence_classes)), ">=", str(K), "*", str(len(df_anonymize)), ": "
          , str(c_dm(equivalence_classes) >= (K * len(df_anonymize))))

    print("CONDITION: C_avg >= 1: ")

    print(str(c_avg(equivalence_classes, df_anonymize, K)), ">= 1: ",
          str(c_avg(equivalence_classes, df_anonymize, K) >= 1))
    """
        for col in df_anonymize.columns:
        print("{}: ".format(col))
        print(np.unique(df_anonymize[col], return_counts=True))
        print("__________________________________________________________")

    """


def algorithm_evaluation_on_k(df, col_type, k_list, column_list):
    global K, data
    data = Data(df, col_type)
    cdm_results = []
    cavg_results = []
    t0 = datetime.now()
    for k in k_list:
        K = k
        print("\n\nK=", K, "\n\n")
        dict_phi = anonymize(data.data_to_anonymize)
        df_anonymize = anonymization(data.data_to_anonymize.data, dict_phi)
        data.data_anonymized = df_anonymize
        equivalence_classes = get_equivalence_classes(df_anonymize, column_list)
        print("\n\nEquivalence Classes:\n\n", equivalence_classes)
        cdm_results.append(c_dm(equivalence_classes))
        cavg_results.append(c_avg(equivalence_classes, df_anonymize, K))
    t1 = datetime.now()
    print("\n\nExecution Time: ", t1 - t0, "\n\n")
    return cdm_results, cavg_results


def plot_evaluations():
    df, cols_to_anonymize = toy_dataset()

    k_list = range(1, 11)
    col_type = {"dim0": Data.NUMERICAL, "dim1": Data.NUMERICAL, "dim2": Data.NUMERICAL}
    column_list = ['dim0_anon', 'dim1_anon', 'dim2_anon']
    cdm_list, cavg_list = algorithm_evaluation_on_k(df, col_type, k_list, column_list)

    print("cdm: ", cdm_list)
    print("cavg: ", cavg_list)

    plt.figure(figsize=(15, 12))
    plt.subplot(2, 1, 1)
    plt.plot(k_list, cdm_list)
    plt.title("Discernability Penalty Metric", fontsize=15)
    plt.xlabel("K", fontsize=15)
    plt.ylabel("$C_{dm}$", fontsize=15)
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(k_list, cavg_list)
    plt.title("Normalized Average Equivalence Class Size Metric", fontsize=15)
    plt.xlabel("K", fontsize=15)
    plt.ylabel("$C_{avg}$", fontsize=15)
    plt.grid()

    plt.show()


if __name__ == "__main__":
    # debug()
    # plot_evaluations()
    # debug_dataset()
    debug_real_dataset()
