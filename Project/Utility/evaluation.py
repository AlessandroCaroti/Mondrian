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
    return round( len(anon_table) / (len(e_c) * k),3)


#def k_tuning()