from typesManager.categoricalManager import categoricalManager
from typesManager.dateManager import DateManager
from typesManager.numericManager import NumericManager
from DGH.dgh import DGH

import pandas as pd
import os


class Data(object):
    NUMERICAL, DATE, CATEGORICAL = list(range(3))  # type of data

    def __init__(self, data, columns_type):  # data is a path or Dataframe??? da decidere--> PATH MEGLIO

        self.data_folder = "Dataset"  # folder containing the csv file
        self.hierarchy_folder = "Hierarchies"  # folder containing the csv files of the dgh

        # data can be either a DataFrame or a file name
        self.dataFrame = data if isinstance(data, pd.DataFrame) else pd.read_csv(os.path.join(self.data_folder, data))

        self.columns_type = columns_type  # dictionary column_name : type_of_data

        self.width_list = {dim: self.compute_width_dim(dim) for dim in
                           self.dataFrame.columns}  # width of all the columns of the original table

        # dgh for each CATEGORICAL column: assuming that the filename is equal to the column name
        self.dgh_list = dict([(dim, DGH(os.path.join(self.hierarchy_folder, dim + ".csv"))) if self.columns_type[
                                                                                                   dim] == Data.CATEGORICAL else (
            dim, None)
                              for dim in self.dataFrame.columns])

    def compute_width_dim(self, dim):

        """
        :param dim: name of the column to compute the width
        :return the width according to the type of data
        """

        if self.columns_type[dim] == Data.NUMERICAL:
            return NumericManager.compute_width(self.dataFrame[dim])

        if self.columns_type[dim] == Data.CATEGORICAL:
            return categoricalManager.compute_width(self.dataFrame[dim])

        if self.columns_type[dim] == Data.DATE:
            return DateManager.compute_width(self.dataFrame[dim])

        raise Exception("column type not valid! Only NUMERICAl, CATEGORICAL and DATE are supported.")
