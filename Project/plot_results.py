from datetime import datetime

from Utility.evaluation import *
from Utility.utility import *
import Utility.data as data
import matplotlib.pyplot as plt
from mondrian import anonymize, anonymization, DATA, K, N_PARTITIONS
import mondrian
import os
import pandas as pd


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

    print("Total time:      ", execution_time)

    columns = df_anonymize.columns.tolist()
    equivalence_classes = get_equivalence_classes(df_anonymize, columns)

    cavg = c_avg(equivalence_classes, df_anonymize, K)

    return cavg, execution_time


def algorithm_evaluation_on_k(dataset, k_list):
    cavg_results = []
    execution_time_list = []

    for k in k_list:
        print("\n\nK={}\n\n".format(k))
        cavg, execution_time = evaluate(dataset, k)

        cavg_results.append(cavg)
        execution_time_list.append(execution_time)

    return cavg_results, execution_time_list


def plot_and_save_evaluations(args, dataset, k_list):
    cavg_results, execution_time_list = algorithm_evaluation_on_k(dataset, k_list)
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
    res.to_csv("Evaluation/Execution_time_" + args.folder_name + "_" + args.dataset_name.split('.')[0] + ".csv",
               index=False, header=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    # print algorithm parameters
    print_args(args)

    k_list = range(2, 100, 2)
    dataset = data.Data(args.folder_name, args.dataset_name, args.columns_type, args.result_name)

    plot_and_save_evaluations(args=args, dataset=dataset, k_list=k_list)
