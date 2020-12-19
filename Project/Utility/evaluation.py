"""
@param e_c: group-bys on quasi-identifier (equivalence classes)
ref: https://www.mdpi.com/2079-9292/9/5/716/htm
"""


def c_dm(e_c):
    # Discernibility Penalty
    # assignment of penalty (cost) to each tuple in the generalized data set.
    # sum of  size( equivalence classes )^2
    tot_penalty = 0
    for E in e_c:
        tot_penalty += len(E) ** 2
    return tot_penalty


#  The normalized average
# measures the quality of the sanitized data by the EC average size.
"""
@param anon_table: number of  anonymized tuples 
@param e_c: group-bys on quasi-identifier (equivalence classes)

@param k: param k of k-anonymity
"""


# TODO: eventualmente da cambiare
def c_avg(e_c, anon_table, k):
    return len(anon_table) / (len(e_c) * k)
