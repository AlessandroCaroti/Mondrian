class Partition(object):

    def __init__(self, partition, width=None, median=None):
        self.width = width  # dictionary of width of the partition for each dimension
        self.median = median  # median of the partition for each dimension
        self.data = partition  # table of the corresponding partition in the tree
