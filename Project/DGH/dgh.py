import csv
from io import StringIO
from DGH.tree import Node, Tree


class DGH:

    def __init__(self, dgh_path):

        """
        Represents multiple hierarchies as a dictionary of trees.

        :param dgh_path:            Path to the file which contains the DGH definition.
        :raises FileNotFoundError:  If the file is not found.
        :raises IOError:            If the file cannot be read.
        """

        self.hierarchy = None
        """
        Dictionary where the values are trees and the keys are the values of the corresponding 
        roots. The dictionary must have only one value, so there must be only one tree
        """

        self.gen_level = None
        """
        Dictionary whose keys are the hierarchies root values and whose values are the hierarchies 
        depths (number of generalization levels).
        """

    def generalize(self, value, gen_level=None):

        """
        Returns the upper lever generalization of a value in the domain.

        :param value:       Value to generalize.
        :param gen_level:   Current level of generalization, where 0 means it's not generalized.
        :return:            The generalized value on the level above, None if it's a root.
        :raises KeyError:   If the value is not part of the domain.
        """

        # Try to find the node:
        if gen_level is None:
            node = self.hierarchy.bfs_search(value)
        else:
            node = self.hierarchy.bfs_search(value, self.gen_level - gen_level)  # Depth.

        if node.parent is None:
            # The value is a hierarchy root:
            return None
        else:
            return node.parent.data


class CsvDGH(DGH):

    def __init__(self, dgh_path):

        super().__init__(dgh_path)

        try:
            with open(dgh_path, 'r') as file:
                for i, line in enumerate(file):

                    try:
                        csv_reader = csv.reader(StringIO(line))
                    except IOError:
                        raise

                    values = next(csv_reader)

                    # If it doesn't exist a hierarchy with this root, add if only it's the first:
                    if i == 0:
                        self.hierarchy = Tree(Node(values[-1]))
                        # Add the number of generalization levels:
                        self.gen_level = len(values) - 1

                    # Only one tree, so if the value is not in the hierarchy it's ignored (unless it's the first)
                    if values[-1] == self.hierarchy:
                        continue

                    # Populate hierarchy with the other values:
                    self._insert_hierarchy(values[:-1], self.hierarchy)

        except FileNotFoundError:
            raise
        except IOError:
            raise

    @staticmethod
    def _insert_hierarchy(values, tree):

        """
        Inserts values, ordered from child to parent, to a tree.

        :param values:  List of values to insert.
        :param tree:    Tree where to insert the values.
        :return:        True if the hierarchy has been inserted, False otherwise.
        """

        leaf_value = values[0]  # get the first value as leaf, the no generalized one

        current_node = tree.root

        for i, value in enumerate(reversed(values)):

            if value in current_node.children:
                # update the list of leaf for each node
                current_node.add_leaf(leaf_value)

                current_node = current_node.children[value]
                continue
            else:
                # Insert the hierarchy from this node
                for v in list(reversed(values))[i:]:
                    current_node.add_child(Node(v))

                    # update the list of leaf for each node
                    current_node.add_leaf(leaf_value)

                    current_node = current_node.children[v]

                # add leaf also to the Leaf itself
                current_node.add_leaf(leaf_value)
                return True

        return False
