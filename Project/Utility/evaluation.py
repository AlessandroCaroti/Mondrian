import os
import Project.mondrian as mondrian

#  The normalized average
# measures the quality of the sanitized data by the EC average size.
"""
@param anon_table: number of  anonymized tuples 
@param e_c: group-bys on quasi-identifier (equivalence classes)

@param k: param k of k-anonymity
"""


def c_avg(n_partition, anon_table, k):
    return round(len(anon_table) / (n_partition * k), 3)


def save_statistics(path, cavg, t0, t1, t2, n_partitions, n_tuple, n_dim, K):
    f = open(os.path.join(path, "statistics_result_K_" + str(K) + ".txt"), "w")
    f.write("\n---------------------------------EVALUATION-STATISTICS-------------------------------------------\n")
    f.write("\nnormalized average equivalence class size metric: {}\n".format(cavg))
    f.write("\nTotal Execution Time: {}\n".format(t2 - t0))
    f.write("\nExecution Time - Computation PHI: {}\n".format(t1 - t0))
    f.write("\nPartition created: {}\n".format(n_partitions))
    f.write("\nSize of the Dataset_synthetic: {}  -  Number of Attribute: {}  -  K: {}".format(n_tuple, n_dim, K))
    f.close()
