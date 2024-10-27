class Node:
    def __init__(self, opt_attr_id, opt_attr_name, depth=0, father=None):
        """
        Parameters
        ----------
        opt_attr_id: int
            The id of the best attribute to split
        opt_attr_name: str
            The name of the best attribute to split
        depth: int, optional
            The depth of the node in the decision tree
        father: Node, optional
            The father of the node
        """
        self.opt_attr_id = opt_attr_id
        self.depth = depth
        self.opt_attr_name = opt_attr_name
        self.child = dict()

        # used in pruning
        self.father = father

    def __call__(self, data):
        """
        Recursively call the child node corresponding to the attribute value of the given data.

        Parameters
        ----------
        data: array-like
            An observation or instance with attributes to be evaluated by the node.

        Returns
        -------
        The result of calling the child node that matches the attribute value of the given data.

        Raises
        ------
        KeyError
            If there is no child for the attribute value in the data.
        """
        if len(self.child) == 0:
            raise KeyError

        return self.child[data[self.opt_attr_id]](data)

    def isRoot(self):
        """
        Check if the node is a root node.

        Returns
        -------
        bool
            True if the node is a root node, False otherwise.
        """
        return self.depth == 0

    def isLeaf(self):
        """
        Check if the node is a leaf node.

        Returns
        -------
        bool
            True if the node is a leaf node, False otherwise.
        """
        return False

    def __repr__(self):
        """
        Return a string representation of the node.

        The string representation includes the attribute used to split the data
        at this node, and the child nodes that the data could be sent to based
        on the attribute values. The string is formatted to show the tree
        structure of the decision tree.

        Returns
        -------
        str
            A string representation of the node.
        """
        str = f'Used Attribute: {self.opt_attr_name}\n'
        for cnt, (cdk, cdv) in enumerate(self.child.items()):
            if cnt == len(self.child) - 1:
                if type(cdv) is Leaf:
                    str += f'{self.depth * "│ "}└ {self.opt_attr_name}{cdk} -> {cdv}'
                    # str += f'\n{self.depth * "│ "}'
                else:
                    str += f'{(self.depth + 1) * "│ "}{self.opt_attr_name}{cdk} -> {cdv}'
                    # str += f'\n{self.depth * "│ "}'
            else:
                str += f'{(self.depth + 1) * "│ "}{self.opt_attr_name}{cdk} -> {cdv}\n'
        return str


class Leaf:
    def __init__(self, label, depth=0):
        """
        Initialize a Leaf node.

        Parameters
        ----------
        label : any
            The label associated with this leaf node.
        depth : int, optional
            The depth of the node in the decision tree (default is 0).
        """
        self.label = label
        self.depth = depth

    def __call__(self, data):
        """
        Predict the label for the given data.

        Parameters
        ----------
        data : array
            The data to predict the label for.

        Returns
        -------
        any
            The label associated with this leaf node.
        """
        return self.label

    def __repr__(self):
        """
        Return a string representation of the Leaf node.

        The string representation includes the label associated with this
        leaf node.

        Returns
        -------
        str
            A string representation of the Leaf node.
        """
        return f'Class: {self.label}'

    def isLeaf(self):
        """
        Check if the node is a leaf node.

        Returns
        -------
        bool
            True if the node is a leaf node, False otherwise.
        """
        return True

    def isRoot(self):
        """
        Check if the node is a root node.

        Returns
        -------
        bool
            True if the node is a root node, False otherwise.
        """
        return self.depth == 0
