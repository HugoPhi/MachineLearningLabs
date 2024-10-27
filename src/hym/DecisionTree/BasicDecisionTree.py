from .node import Node, Leaf
import copy
import numpy as np


class DecisionTree:

    def __init__(self, data, label, attr_dict, key2id=dict(), depth=0, valid=None, valid_label=None, pruning='none', id2name=dict()):
        # self attributes
        """
        Initialize DecisionTree.

        Parameters
        ----------
        data : 2d array
            training data
        label : 1d array
            labels of the training data
        attr_dict : dict
            {attr_name: [attr_val1, attr_val2, ...]}
        key2id : dict
            {attr_name: attr_id}
        depth : int
            depth of the tree
        valid : 2d array
            validation data
        valid_label : 1d array
            labels of the validation data
        pruning : str
            way of pruning
        id2name : dict
            {attr_id: attr_name}
        """
        self.data = data
        self.label = label
        self.attr_dict = attr_dict
        self.key2id = key2id
        self.depth = depth
        self.valid = valid
        self.valid_label = valid_label
        self.pruning = pruning
        self.id2name = id2name

        # tree
        self.tree = None

    def fit(self):
        """
        Builds and fits the decision tree model using the training data.

        This method constructs the decision tree model based on the provided
        training data, labels, and attributes. It uses a recursive approach
        to build the tree structure by selecting optimal attributes at each
        node. The fit process also considers pruning strategies if specified.

        Returns
        -------
        Node
            The root node of the constructed decision tree.
        """
        self.tree = self.build(
            data=self.data,
            label=self.label,
            attr_dict=self.attr_dict,
            key2id=self.key2id,
            depth=self.depth,
            valid=self.valid,
            valid_label=self.valid_label,
            father=None,
            pruning=self.pruning)
        return self.tree

    def __call__(self, data):
        """
        Makes predictions on the input data using the constructed decision tree model.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The input data to make predictions on.

        Returns
        -------
        array-like of shape (n_samples,)
            Predicted labels for the input data.
        """
        if data.ndim == 1:
            return self.tree(data)
        else:
            return np.array([self.tree(x) for x in data])

    def build(self, data, label, attr_dict, key2id=dict(), depth=0, valid=None, valid_label=None, father=None, pruning='none'):
        """
        Recursively builds the decision tree model.

        This method constructs the decision tree by recursively selecting 
        the optimal attribute at each node and partitioning the data 
        according to attribute values. It handles leaf node creation 
        when data is pure, attributes are exhausted, or data is uniform 
        with respect to an attribute. Pruning strategies can be applied 
        to optimize the tree structure.

       Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The training data to build the tree from.
        label : array-like of shape (n_samples,)
            The labels corresponding to the training data.
        attr_dict : dict
            A dictionary mapping attribute names to possible values.
        key2id : dict, optional
            A dictionary mapping attribute names to their indices.
        depth : int, optional
            The current depth of the tree.
        valid : array-like, optional
            Validation data used for pruning.
        valid_label : array-like, optional
            Validation labels used for pruning.
        father : Node, optional
            The parent node of the current subtree.
        pruning : str, optional
            The pruning strategy to apply ('none', 'pre', or 'post').

        Returns
        -------
        Node
            The root node of the constructed subtree.
        """
        if father is None:
            key2id = {key: idx for idx, key in enumerate(attr_dict.keys())}
        if (label[0] == label).all():
            return Leaf(self.id2name[label[0]], depth)

        if len(attr_dict) == 0:
            return Leaf(self.id2name[np.bincount(label).argmax()], depth)

        aflag = True
        for attr_ids in [key2id[k] for k in attr_dict.keys()]:
            if (data[0, attr_ids] != data[:, attr_ids]).any():
                aflag = False
                break

        if aflag is True:
            return Leaf(self.id2name[np.bincount(label).argmax()], depth)

        opt_attr_id, attr_vals, attr_dict_without_opt_attr = self.opt_attr(data, label, attr_dict, key2id)
        opt_attr_name = opt_attr_id
        opt_attr_id = key2id[opt_attr_id]

        tree = Node(
            opt_attr_id=opt_attr_id,
            opt_attr_name=opt_attr_name,
            father=father,
            depth=depth)

        tree.father = tree

        if not pruning == 'none':
            pre_accuracy = np.mean(np.bincount(label).argmax() == label)

        for attr_val in attr_vals:
            label_of_same_attrval = label[data[:, opt_attr_id] == attr_val]

            if (len(label_of_same_attrval) == 0):
                tree.child[attr_val] = Leaf(self.id2name[np.bincount(label).argmax()], depth + 1)
            else:
                tree.child[attr_val] = Leaf(self.id2name[np.bincount(label_of_same_attrval).argmax()], depth + 1)

        after_precision = 0
        if pruning == 'pre':  # Strategy: Maximize metric
            after_precision = self.pruning_metric(valid, valid_label, tree)
            if not pre_accuracy < after_precision:
                return Leaf(self.id2name[np.bincount(label).argmax()], depth)

        for attr_val in attr_vals:
            data_of_same_attrval = data[data[:, opt_attr_id] == attr_val]
            label_of_same_attrval = label[data[:, opt_attr_id] == attr_val]
            if pruning == 'pre':
                valid_of_same_attrval = valid[valid[:, opt_attr_id] == attr_val]
                valid_label_of_same_attrval = valid_label[valid[:, opt_attr_id] == attr_val]
            else:
                valid_of_same_attrval = valid
                valid_label_of_same_attrval = valid_label

            if (len(label_of_same_attrval) == 0):
                tree.child[attr_val] = Leaf(self.id2name[np.bincount(label).argmax()], depth + 1)
                continue

            tree.child[attr_val] = self.build(
                data=data_of_same_attrval.copy(),
                label=label_of_same_attrval.copy(),
                attr_dict=attr_dict_without_opt_attr.copy(),
                key2id=key2id,
                valid=valid_of_same_attrval.copy(),
                valid_label=valid_label_of_same_attrval.copy(),
                father=tree,
                pruning=pruning,
                depth=depth + 1)

        if pruning == 'post' and tree.isRoot():
            tree = self.post_pruning(valid, valid_label, tree)

        return tree

    def pruning_metric(self, data, label, tree):
        '''
        Pre-Pruning Strategy of Decision Tree, Maximize metric here, use accuracy.   
        That is, you should Maximize the accuracy on validation set while pruning.

        Parameters
        ----------
        data : array-like
            The data to be classified.
        label : array-like
            The labels of the data.
        tree : Node
            The tree to be pruned.

        Returns
        -------
        float
            The accuracy on validation set.
        '''

        res = []
        for x in data:
            res.append(tree(x))
        res = np.array(res)
        label = np.array([self.id2name[label[x]] for x in label])
        return np.mean(res == label)

    def post_pruning(self, valid, valid_label, tree_node, root=None):
        '''
        Post-Pruning Strategy of Decision Tree, Maximize metric here, use accuracy.  
        That is, you should Maximize the accuracy on validation set while pruning.

        Parameters
        ----------
        data : array-like
            The data to be classified.
        label : array-like
            The labels of the data.
        tree : Node
            The tree to be pruned.

        Returns
        -------
        Node
            The pruned tree
        '''

        root = root
        all_children_are_leaf = True

        if tree_node.isRoot():
            root = tree_node

        for key, child in tree_node.child.items():
            if not child.isLeaf():
                tree_node.child[key] = self.post_pruning(valid, valid_label, child, root)
                all_children_are_leaf = False

        if all_children_are_leaf:
            pre_precision = self.pruning_metric(valid, valid_label, root)
            tree_copy = copy.deepcopy(tree_node)

            tree_node = Leaf(self.id2name[np.bincount(valid_label).argmax()], tree_node.depth)
            post_precision = self.pruning_metric(valid, valid_label, root)
            if pre_precision > post_precision:
                tree_node = tree_copy
            else:
                return tree_node

        return tree_node

    def attr_selection_metric(self, data, label, attr, attr_val):
        """
        Function to Calculate the best attribute. Based on information gain here.

        This function computes the information gain of a given attribute
        by comparing the entropy of the labels before and after splitting
        the data based on the attribute values.

        Parameters
        ----------
        data : 2D array
            The dataset containing features and samples.
        label : 1D array
            The labels corresponding to the samples in the dataset.
        attr : int
            The index of the attribute for which the metric is calculated.
        attr_val : list
            The list of possible values for the attribute.

        Returns
        -------
        float
            The information gain of the attribute.
        """

        def Ent(label):
            prob = np.bincount(label) / len(label)
            res = np.array([p * np.log2(p) if p != 0 else 0 for p in prob])
            return -np.sum(res)

        gain = Ent(label)
        for val in attr_val:
            label_temp = label[data[:, attr] == val]
            if len(label_temp) == 0:
                continue
            gain -= len(label_temp) / len(label) * Ent(label_temp)
        return gain

    def opt_attr(self, data, label, attr_dict, key2id):
        """
        Selects the optimal attribute for splitting the data based on function: `attr_selection_metric`.

        This method computes the information gain for each attribute in the
        provided attribute dictionary and selects the attribute with the highest
        information gain as the optimal attribute for splitting the data. It returns
        the name of the selected attribute, its possible values, and an updated
        attribute dictionary excluding the selected attribute.

        Parameters
        ----------
        data : 2D array
            The dataset containing features and samples.
        label : 1D array
            The labels corresponding to the samples in the dataset.
        attr_dict : dict
            A dictionary mapping attribute names to possible values.
        key2id : dict
            A dictionary mapping attribute names to their indices.

        Returns
        -------
        tuple
            A tuple containing the name of the selected attribute, a list of its
            possible values, and an updated attribute dictionary without the selected
            attribute.
        """
        attr = np.argmax([self.attr_selection_metric(data, label, key2id[key], attr_val) for key, attr_val in attr_dict.items()])
        attr_val = list(attr_dict.values())[attr]
        aattr_dict = {k: v for j, (k, v) in enumerate(attr_dict.items()) if j != attr}  # remove the selected attribute

        attr = list(attr_dict.keys())[attr]  # the name of the selected attribute
        return attr, attr_val, aattr_dict