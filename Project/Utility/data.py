from Project.typesManager.categoricalManager import CategoricalManager
from Project.typesManager.dateManager import DateManager
from Project.typesManager.numericManager import NumericManager
from Project.Partition.partition import Partition
from Project.DGH.dgh import CsvDGH

import pandas as pd
import os


class Data(object):
    EI, SD, NUMERICAL, DATE, CATEGORICAL = list(range(5))  # type of data: Explicit Identifiers, Sensitive data and
                                                            # Quasi-Identifiers: can be NUMBER, CATEGORICAL or DATE

    def __init__(self, data, columns_type):

        self.columns_type = columns_type  # dictionary column_name : type_of_data
        self.data_folder = "Dataset"  # folder containing the csv file
        self.hierarchy_folder = "Hierarchies"  # folder containing the csv files of the dgh

        # data can be either a DataFrame or a file name
        self.dataFrame = data if isinstance(data, pd.DataFrame) else pd.read_csv(os.path.join(self.data_folder, data))

        # dgh for each CATEGORICAL column: assuming that the filename is equal to the column name
        dgh_list = []
        for dim in self.dataFrame.columns:
            if self.columns_type[dim] == Data.CATEGORICAL:
                dgh_list.append((dim, CsvDGH(os.path.join(self.hierarchy_folder, dim + ".csv"))))

        self.dgh_list = dict(dgh_list)

        # select the columns to anonymize
        col_to_anonymize = []
        for col, type in self.columns_type.items():
            if Data.is_qi(type):
                col_to_anonymize.append(col)

        # width of all the QI of the original table
        self.width_list = {dim: self.init_width_dim(dim) for dim in col_to_anonymize}

        # median of all the QI of the original table
        self.median_list = {dim: self.init_median_dim(dim) for dim in col_to_anonymize}

        # create a Partition with the table to anonymize
        self.data_to_anonymize = Partition(self.dataFrame[col_to_anonymize], self.width_list, self.median_list)

        self.data_anonymized = None
    def init_width_dim(self,dim):

        """
        Initialize the width given a dim
        :param dim: name of the column to compute the width
        :return the width according to the type of data
        """

        if self.columns_type[dim] == Data.CATEGORICAL:
            tree = next(iter(self.dgh_list[dim].hierarchies.values())) # first element of the dictionary
            return len(tree.root.leaf) # the initial width is the root Node of the DGH

        if self.columns_type[dim] == Data.DATE:
            p = Partition(self.dataFrame)
            return DateManager.width(p, dim)

        if self.columns_type[dim] == Data.NUMERICAL:
            p = Partition(self.dataFrame)
            return NumericManager.width(p, dim)

        raise Exception("column type not valid! Only NUMERICAl, CATEGORICAL and DATE are supported.")

    def init_median_dim(self,dim):

        """
        Initialize the median given a dim
        :param dim: name of the column to compute the width
        :return the width according to the type of data
        """

        if self.columns_type[dim] == Data.CATEGORICAL:
            item = next(iter(self.dgh_list[dim].hierarchies.values()))
            return item.root # the initial median is the root Node of the DGH

        if self.columns_type[dim] == Data.DATE:
            p = Partition(self.dataFrame)
            return DateManager.median(p, dim)

        if self.columns_type[dim] == Data.NUMERICAL:
            p = Partition(self.dataFrame)
            return NumericManager.median(p, dim)

        raise Exception("column type not valid! Only NUMERICAl, CATEGORICAL and DATE are supported.")

    @staticmethod
    def is_qi(type):
        '''
        Given a type return if it's a QI
        '''

        return not (type == Data.EI or type == Data.SD)