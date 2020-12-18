import collections
from numbers import Number

import numpy as np
import pandas as pd
from numpy import random
from pandas.api.types import is_numeric_dtype


def chose_dimension(dimensions, step):
    dim_pos = int(step % len(dimensions))
    return dimensions[dim_pos]


def merge_dictionary(dict1, dict2):
    return {**dict1, **dict2}


def compute_phi(partition):
    np_partition = partition.to_numpy()
    summary = []

    for col in range(np_partition.shape[1]):
        _max = np.max(np_partition[col, :])
        _min = np.min(np_partition[col, :])
        col_summary = "[" + str(_min) + " - " + str(_max) + "]"
        if _min == _max:
            col_summary = str(_min)
        summary.append(col_summary)

    phi = {}
    for idx in partition.index:
        phi[idx] = summary
        # phi[partition.iloc(idx)] = summary
    return phi


def find_median(partition, dim):
    if is_numeric_dtype(partition[dim]):
        return partition[dim].median()

    # todo: gestire le colonne che non sono numeri
    return None


def split_partition(partition, dim, split_val):
    if isinstance(split_val, Number):
        left_p = partition[partition[dim] >= split_val]
        right_p = partition[partition[dim] < split_val]
    else:  # TODO: da gestire il caso in cui non sia un numero
        left_p, right_p = None, None

    return left_p, right_p


def allowable_cut(partition, dim, split_val, k):
    value_list = partition[dim].unique()
    if len(value_list) <= 1:
        return False
    if len(np.where(value_list < split_val)[0]) < k:
        return False
    return True


def anonymize(partition, columns, step, k):
    dim = chose_dimension(columns, step)
    mean = find_median(partition, dim)

    # If not allowed multidimensional cut for partition
    if not allowable_cut(partition, dim, mean, k):
        return compute_phi(partition)  # return phi: partition -> summary

    lhs, rhs = split_partition(partition, dim, mean)

    phi_merger = merge_dictionary(anonymize(lhs, columns, step + 1, k),
                                  anonymize(rhs, columns, step + 1, k))

    return phi_merger


def debug():
    # GENERATE A TOY DATASET
    n_sample = 20
    n_cols = 2
    cols = []

    for i in range(n_cols):
        cols.append("dim_" + str(i))

    data = random.randint(0, 10, (n_sample, n_cols))
    df = pd.DataFrame(data, columns=cols)

    # ANONYMIZE DATA
    dict_phi = anonymize(df, cols, 0, 3)

    od = collections.OrderedDict(sorted(dict_phi.items()))
    for pair in od.items():
        print(pair)


if __name__ == "__main__":
    debug()

# TODO: implementare una choise_dim() migliore, che ritorna la dimensione con norma maggiore                ()
# TODO: implementare la scelta della MEDIANA anche per valori che non sono numeri(da discutere come)        ()
# TODO: implementare la SPLIT anche per valori che non sono numeri(da discutere come)                       ()
# TODO: implemetare summary statistic anche per valori che non sono numeri(da discutere come)               ()
# TODO: implementare la parte in cui dato il dizionario, si anonimizza la tabella originale                 ()
# TODO: implementare metodi di valutazione per l'anonimizzazione ottenuta                                   ()
# TODO: completare la creazione di un database                                                              (ALE)
