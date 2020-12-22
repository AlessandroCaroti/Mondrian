from typesManager.categoricalManager import categoricalManager
from typesManager.dateManager import dateManager
from typesManager.numericManager import numericManager
from DGH.dgh import DGH

import pandas as pd
import os

class Data():

    NUMERICAL, CATEGORICAL, DATE = list(range(3)) # type of data

    def __init__(self, fileName, columns_type, ): # data is a path or Dataframe??? da decidere

        self.data_folder = "Dataset" # folder containing the csv file
        self.hierarchy_folder = "Hierarchies" # folder containing the csv files of the dgh
        self.dataFrame = pd.read_csv(os.path.join(self.data_folder, fileName))

        self.columns_type = columns_type # dictionary column_name : type_of_data

        self.width_list = {dim: self.compute_width_dim(dim) for dim in self.dataFrame.columns} # width of all the columns of the original table
        self.dgh_list = {} # dgh for each CATEGORICAL column. assuming that the filename is equal to the column name

    def compute_width_dim(self, dim):

        '''
        :param dim: name of the column to compute the width
        :return the width according to the type of data
        '''

        if self.columns_type[dim] == Data.NUMERICAL:
            return numericManager.compute_width(self.dataFrame[dim])

        if self.columns_type[dim] == Data.CATEGORICAL:
            return categoricalManager.compute_width(self.dataFrame[dim])

        if self.columns_type[dim] == Data.DATE:
            return dateManager.compute_width(self.dataFrame[dim])

        raise Exception("column type not valid! Only NUMERICAl, CATEGORICAL and DATE are supported.")