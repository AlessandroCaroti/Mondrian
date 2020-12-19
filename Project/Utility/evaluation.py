# TODO : quality measure is based on the size of the equivalence
#  classes E in V . Cdm assigns to each tuple t in V a penalty,
#  which is determined by the size of the equivalence class containing t.

"""
@param e_c: group-bys on quasi-identifier (equivalence classes)

"""


def c_dm(e_c):
    # Discernibility Penalty
    # assignment of penalty (cost) to each tuple in the generalized data set.
    # sum of  size( equivalence classes )^2
    tot_penalty = 0
    for E in e_c:
        tot_penalty += len(E)**2
    return tot_penalty




#  As an alternative, we also propose the normalized average
#  equivalence class size metric (C_avg).

"""
@param e_c: group-bys on quasi-identifier (equivalence classes)
@param num_tuples: number of tuples in the table
@param k: param k of k-anonymity
"""


# TODO: eventualmente da cambiare quando abbiamo
#  la tabella finita
def c_avg(e_c, num_tuples, k):
    return num_tuples / (e_c * k)
