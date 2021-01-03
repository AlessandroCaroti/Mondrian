import argparse

def get_parser():

    parser = argparse.ArgumentParser(description="Mondrian algorithm's parameters")

    # add arguments...

    # as default dataset the syntethic one created by us
    parser.add_argument("-dataset_name", help="name of csv file containing the data", default="mainDB_10000.csv")
    parser.add_argument("-columns_type", help="name of csv file containing for each row the column_name and the type " +
                                              "(EI, SD, NUMERICAL, DATE, CATEGORICAL)", default="columns_type.csv")

    parser.add_argument("-K", help="parameter K", type=int, default=10)
    parser.add_argument("-result_name", help="name of the file will contain the resulting data", default="anonymized.csv")

    # if False the result may contains columns with max generalizations (in particular CATEGORICAL)
    parser.add_argument("-use_all_col", help="if True it guarantees that each column is used at least once to split", default=False)
    parser.add_argument("-save_statistics", help="whether to save time execution and equivalence classes", type=bool, default=True)
    parser.add_argument("-show_statistics", help="whether to print time execution and equivalence classes", type=bool, default=True)

    return parser

def print_args(args):

    print()
    print("===================")
    print("    Parameters")
    print("-------------------")

    print("k: {}".format(args.K))
    print("Original data: {}".format(args.dataset_name))
    print("Result data: {}".format(args.result_name))

    print("===================")
    print()

def save_statistics():

    """
    f = open("results/statistics_result_DB.txt", "w")
    f.write("\n---------------------------------EVALUATION-STATISTICS-------------------------------------------\n")
    f.write("\nDiscernability Penalty Metric: {}\n".format(cdm))
    f.write("\nDiscernability Penalty Metric: {}\n".format(cavg))
    f.write("\nTotal Execution Time: {}\n".format(t2 - t0))
    f.write("\nExecution Time - Computation PHI: {}\n".format(t1 - t0))
    f.write("\nPartition created: {}\n".format(total_partition))
    f.write("\nSize of the Dataset: {}  -  Number of Attribute: {}  -  K: {}".format(len_dataset, n_dim, K))
    f.close()

    """
    # SAVE ALL STATISTICS IN THE FOLDER RESULTS

