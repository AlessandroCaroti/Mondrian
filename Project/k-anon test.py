from numbers import Number
from pandas.api.types import is_numeric_dtype


def chose_dimension(dimensions):
    return ""


def compute_phi(partition, dim):
    return 0


def find_median(partition, dim):
    if is_numeric_dtype(partition[dim]):
        return partition[dim].median()

    # todo: gestire le colonne che non sono numeri
    return None


def split_partition(partition, dim, split_val):
    if isinstance(split_val, Number):
        left_p = partition[partition[dim] >= split_val]
        right_p = partition[partition[dim] < split_val]
    else:   # TODO: da gestire il caso in cui non sia un numero
        left_p, right_p = None, None

    return left_p, right_p


def anonymize(partition, columns):
    # If nor allowed multidimensional cut for partition
    #   return phi: partition -> summary

    dim = chose_dimension(columns)
    mean = find_median(partition, dim)
    lhs, rhs = split_partition(partition, dim, mean)

    return [anonymize(lhs, columns),
            anonymize(rhs, columns)]
