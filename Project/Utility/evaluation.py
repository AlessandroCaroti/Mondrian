import os

"""
@param e_c: group-bys on quasi-identifier (equivalence classes)
ref: https://www.mdpi.com/2079-9292/9/5/716/html
"""


# A group of records that are indistinguishable from each other is
# often referred to as an equivalence class.

def get_equivalence_classes(anon_dataset, columns_list):
    return anon_dataset.groupby(by=columns_list).size().reset_index(name='counts')


def c_dm(e_c):
    # Discernibly Penalty
    # assignment of penalty (cost) to each tuple in the generalized data set.
    # sum of  size( equivalence classes )^2
    tot_penalty = 0
    for e in e_c['counts']:
        tot_penalty += e ** 2
    return tot_penalty


#  The normalized average
# measures the quality of the sanitized data by the EC average size.
"""
@param anon_table: number of  anonymized tuples 
@param e_c: group-bys on quasi-identifier (equivalence classes)

@param k: param k of k-anonymity
"""

def c_avg(e_c, anon_table, k):
    return round(len(anon_table) / (len(e_c) * k), 3)

def save_statistics(path, cdm, cavg, t0, t1, t2, n_partitions, n_tuple, n_dim, K):

    f = open(os.path.join(path, "statistics_result.txt"), "w")
    f.write("\n---------------------------------EVALUATION-STATISTICS-------------------------------------------\n")
    f.write("\nDiscernability Penalty Metric: {}\n".format(cdm))
    f.write("\nDiscernability Penalty Metric: {}\n".format(cavg))
    f.write("\nTotal Execution Time: {}\n".format(t2 - t0))
    f.write("\nExecution Time - Computation PHI: {}\n".format(t1 - t0))
    f.write("\nPartition created: {}\n".format(n_partitions))
    f.write("\nSize of the Dataset: {}  -  Number of Attribute: {}  -  K: {}".format(n_tuple, n_dim, K))
    f.close()





