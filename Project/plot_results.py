from datetime import datetime

from Utility.evaluation import *
from Utility.utility import *
import Utility.data as data
import matplotlib.pyplot as plt
from mondrian import anonymize, anonymization
import mondrian
import os
import pandas as pd
import csv


def convert_time(str_time):
    time_delta = pd.Timedelta(str_time)
    tsec = time_delta.total_seconds()

    return tsec / 60


def evaluate(dataset, k):
    mondrian.init(data=dataset, k=k)

    print("START MONDRIAN")

    t0 = datetime.now()
    # ANONYMIZE QUASI-IDENTIFIERS: find phi function
    dict_phi = anonymize(dataset.partition_to_anonymize, False)

    # anonymize the partition with all the dataset
    df_anonymize = anonymization(dataset.partition_to_anonymize.data, dict_phi)

    t2 = datetime.now()

    execution_time = t2 - t0
    print("END MONDRIAN\n")

    # save result in a file
    dataset.data_anonymized = df_anonymize
    dataset.save_anonymized(k)

    print("Total time:      ", execution_time)

    cavg = c_avg(mondrian.N_PARTITIONS, df_anonymize, k)

    return cavg, execution_time


def algorithm_evaluation_on_k(dataset, k_list, backup_file_path):
    cavg_results = []
    execution_time_list = []
    backup_folder_path = r"Evaluation/backup"

    for k in k_list:
        print("\n\nK={}\n\n".format(k))

        cavg, execution_time = evaluate(dataset, k)
        execution_time = convert_time(execution_time)

        cavg_results.append(cavg)
        execution_time_list.append(execution_time)

        if not os.path.isdir(backup_folder_path):
            os.makedirs(backup_folder_path)

        if not os.path.isfile(backup_file_path):
            pd.DataFrame(columns=['K', 'C_Avg', 'Execution Time (minutes)']).to_csv(backup_file_path,
                                                                                    index=False, header=True)

        with open(backup_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([k, cavg, execution_time])

    return cavg_results, execution_time_list


def plot_and_save_evaluations(args, dataset, k_list, backup_file_path):
    cavg_results, execution_time_list = algorithm_evaluation_on_k(dataset, k_list, backup_file_path)
    print(cavg_results, '\n', k_list)
    plt.plot(k_list, cavg_results)
    plt.title("Normalized Average Equivalence Class Size Metric", fontsize=15)
    plt.xlabel("K", fontsize=15)
    plt.ylabel("$C_{avg}$", fontsize=15)
    plt.grid()

    if not os.path.isdir("Evaluation/graphics"):
        os.makedirs("Evaluation/graphics")

    plt.savefig("Evaluation/graphics/Graphic_" + args.folder_name + "_" + args.dataset_name.split('.')[0], dpi=100)
    plt.show()
    res = pd.DataFrame()
    res['K'] = k_list
    res['Execution_time'] = execution_time_list
    res['Cavg'] = cavg_results
    res.to_csv("Evaluation/Execution_time_" + args.folder_name + "_" + args.dataset_name.split('.')[0] + ".csv",
               index=False, header=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    # print algorithm parameters
    print_args(args)
    backup_file_path = r"Evaluation/backup/backup_result_" + args.dataset_name.split('.')[0] + ".csv"

    k_list = [2] + list(range(5, 105, 5))
    if os.path.isfile(backup_file_path):
        backup = pd.read_csv(backup_file_path)
        """Se c'è il backup riparte dal k che era rimasto"""
        i = 0
        for l1, l2 in zip(backup['K'].tolist(), k_list):
            if l1 != l2:
                break
            i += 1
        k_list = k_list[i:]

    dataset = data.Data(args.folder_name, args.dataset_name, args.columns_type, args.result_name)

    plot_and_save_evaluations(args=args, dataset=dataset, k_list=k_list, backup_file_path=backup_file_path)
