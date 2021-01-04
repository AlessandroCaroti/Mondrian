import typesManager.dateManager as dm
import typesManager.numericManager as nm
import typesManager.categoricalManager as cm
from Utility.type import Type


class Partition():

    def __init__(self, partition, col_type, width=None, median=None):

        self.width = width  # dictionary of width of the partition for each dimension
        self.median = median  # median of the partition for each dimension
        self.col_type = col_type  # dict name_column : type
        self.data = partition  # table of the corresponding partition in the tree

    def compute_width(self, dim):
        if self.col_type[dim] == Type.NUMERICAL.value:
            width = nm.NumericManager.width(self, dim)

        if self.col_type[dim] == Type.DATE.value:
            width = dm.DateManager.width(self, dim)

        if self.col_type[dim] == Type.CATEGORICAL.value:
            width = cm.CategoricalManager.width(self, dim)

        return width

    def find_median(self, dim):
        if self.col_type[dim] == Type.NUMERICAL.value:
            median = nm.NumericManager.median(self, dim)

        if self.col_type[dim] == Type.DATE.value:
            median = dm.DateManager.median(self, dim)

        if self.col_type[dim] == Type.CATEGORICAL.value:
            median = cm.CategoricalManager.median(self, dim)

        return median

    def split_partition(self, dim, split_val):
        if self.col_type[dim] == Type.NUMERICAL.value:
            left, right = nm.NumericManager.split(self, dim, split_val)
            return [left, right]

        if self.col_type[dim] == Type.DATE.value:
            left, right = dm.DateManager.split(self, dim, split_val)
            return [left, right]

        if self.col_type[dim] == Type.CATEGORICAL.value:
            partition_list = cm.CategoricalManager.split(self, dim, split_val)
            return partition_list

    def compute_phi(self, dim):

        if self.col_type[dim] == Type.NUMERICAL.value:
            return nm.NumericManager.summary_statistic(self, dim)

        if self.col_type[dim] == Type.DATE.value:
            return dm.DateManager.summary_statistic(self, dim)

        if self.col_type[dim] == Type.CATEGORICAL.value:
            return cm.CategoricalManager.summary_statistic(self, dim)

    def update_width(self, dim):

        width = self.compute_width(dim)

        # update width in dict
        self.width[dim] = width

    def update_median(self, dim):
        median = self.find_median(dim)

        # update median in dict
        self.median[dim] = median

    def get_width(self, dim):
        return self.width[dim]

    def get_median(self, dim):
        return self.median[dim]