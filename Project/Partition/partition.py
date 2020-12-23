
class Partition():

    def __init__(self, partition, width, median):

        self.width = width # dictionary of width of the partition for each categorical dimension
        self.median = median # median of the partition for each categorical dimension
        self.data = partition # table of the corresponding partition in the tree



