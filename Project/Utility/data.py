
from Utility.partition import Partition
from DGH.dgh import CsvDGH
from Utility.type import Type

import pandas as pd
import numpy as np
import os


class Data(object):

    def __init__(self, data, columns_type, result_name):

        self.data_folder = "Dataset"  # folder containing the csv file
        self.hierarchy_folder = "Hierarchies"  # folder containing the csv files of the dgh
        self.result_folder = "Results"  # folder containing the resulting files
        self.result_name = result_name  # name used to save the result

        # dictionary column_name : type_of_data, columns_type is a file name or dict
        self.dim_QI, self.dim_SD = self.get_columns_type(columns_type)

        # data can be either a DataFrame or a file name
        self.dataFrame = data if isinstance(data, pd.DataFrame) else pd.read_csv(os.path.join(self.data_folder, data))

        # dgh for each CATEGORICAL column: assuming that the filename is equal to the column name
        dgh_list = []

        # load the generalizations
        for dim, type in self.dim_QI.items():
            if type == Type.CATEGORICAL.value:
                dgh_list.append((dim, CsvDGH(os.path.join(self.data_folder, self.hierarchy_folder, dim + ".csv"))))

        self.dgh_list = dict(dgh_list)

        # select the columns to anonymize
        col_to_anonymize = []
        for col, type in self.dim_QI.items():
            col_to_anonymize.append(col)

        # width of all the QI of the original table
        self.width_list = {dim: self.init_width_dim(dim) for dim in col_to_anonymize}

        # median of all the QI of the original table
        self.median_list = {dim: self.init_median_dim(dim) for dim in col_to_anonymize}

        # create a Partition with the table to anonymize
        self.partition_to_anonymize = Partition(self.dataFrame[col_to_anonymize], self.dim_QI, self.width_list, self.median_list)

        # contains only the columns defined as Sensitive data
        self.data_SD = self.dataFrame[self.dim_SD]

        self.data_anonymized = None

    def init_width_dim(self, dim):

        """
        Initialize the width given a dim
        :param dim: name of the column to compute the width
        :return the width according to the type of data
        """

        if self.dim_QI[dim] == Type.CATEGORICAL.value:
            return len(np.unique(self.dataFrame[dim]))  # the initial width is the unique values of the entire data

        return Partition(self.dataFrame, self.dim_QI).compute_width(dim)

    def init_median_dim(self, dim):

        """
        Initialize the median given a dim
        :param dim: name of the column to compute the width
        :return the width according to the type of data
        """

        if self.dim_QI[dim] == Type.CATEGORICAL.value:
            item = self.dgh_list[dim].hierarchy

            return item.root.find_minimal_node(np.unique(self.dataFrame[dim]))  # the initial median is the root Node of the DGH

        return Partition(self.dataFrame, self.dim_QI).find_median(dim)


    def get_columns_type(self, file):
        """
        Given a filename.csv with rows like : name_column, type.
        Return a dict as keys the column name and as values the types (Quasi-identifier).
        Return a list containing column name (Sensitive data).
        If parameter is already a dict the function return it.
        """
        if isinstance(file, dict):
            return file

        QI = dict()
        SD = []

        dataframe = pd.read_csv(os.path.join(self.data_folder, file)).to_dict('index')

        for index, col_type in dataframe.items():

            # Primary keys are ignored
            if col_type["Type"] == Type.EI.value:
                continue

            # Sensitive data added to list (no need to know other info about them)
            if col_type["Type"] == Type.SD.value:
                SD.append(col_type["Column_name"])
                continue

            # Quasi-identifier added to dict
            QI[col_type["Column_name"]] = col_type["Type"]

        return QI, SD

    def save_anonymized(self):
        """
        Save a csv file containing the anonymized QI and SD
        """

        if self.data_anonymized is None:
            Exception("Dataset not anonymized yet!")

        df_merged = pd.concat([self.data_anonymized, self.data_SD], axis=1, sort=False)
        df_merged.to_csv(os.path.join(self.get_path_results(), self.result_name))


    def get_path_results(self):

        return os.path.join(self.data_folder, self.result_folder)