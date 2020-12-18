from numbers import Number

from pandas.api.types import is_numeric_dtype


def chose_dimension(dimensions, step):
    dim_pos = int(step % len(dimensions))
    return dimensions[dim_pos]


def compute_phi(partition):
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
    else:  # TODO: da gestire il caso in cui non sia un numero
        left_p, right_p = None, None

    return left_p, right_p


def allowable_cut(partition, dim, split_val, k):
    # TODO: test if is allowable multidimensional cut for partition
    value_list = partition[dim].unique()


def anonymize(partition, columns, step, k):
    # If not allowed multidimensional cut for partition
    #   return phi: partition -> summary

    dim = chose_dimension(columns, step)
    mean = find_median(partition, dim)
    if not allowable_cut(partition, dim, mean, k):
        return compute_phi(partition)

    lhs, rhs = split_partition(partition, dim, mean)

    return [anonymize(lhs, columns, step + 1, k),
            anonymize(rhs, columns, step + 1, k)]
