from .BasicDecisionTree import DecisionTree
import numpy as np
from .utils import get_datas, discretize


class ID3(DecisionTree):
    def __init__(self, df, max_depth=0, valid_rate=0., valid_ix=[], pruning='none', random_state=42):
        """
        Initialize ID3, data will be shuffled.

        Parameters
        ----------
        df : DataFrame
            the training data
        max_depth : int
            maximum depth of the tree
        valid_rate : float
            rate of validation data
        valid_ix : list
            indices of validation data
        pruning : str
            way of pruning
        random_state : int
            random seed
        """
        data, label, attr_dict, id2name = get_datas(df)
        train_ix = np.setdiff1d(np.arange(data.shape[0]), valid_ix)

        # shuffle
        np.random.seed(random_state)
        shuffle_ix = np.random.permutation(len(data))
        data = data[shuffle_ix]
        label = label[shuffle_ix]

        if valid_rate != 0:
            SPLIT = int(data.shape[0] * valid_rate)
            self.valid = data[:-SPLIT]
            self.valid_label = label[:-SPLIT]
            self.data = data[:SPLIT]
            self.label = label[:SPLIT]
        else:
            self.valid = np.array([data[ix] for ix in valid_ix])
            self.valid_label = np.array([label[ix] for ix in valid_ix])
            self.data = np.array([data[ix] for ix in train_ix])
            self.label = np.array([label[ix] for ix in train_ix])

        self.id2name = id2name
        self.pruning = pruning

        super().__init__(
            data=self.data,
            label=self.label,
            attr_dict=attr_dict,
            valid=self.valid,
            valid_label=self.valid_label,
            pruning=self.pruning,
            id2name=self.id2name)

    def datas(self):
        """
        Return a dictionary containing data and labels in the following format:
        {
            'train_data': 2d array,
            'train_label': 1d array,
            'valid_data': 2d array,
            'valid_label': 1d array,
            'pruning method': str,
            'id2name': dict
        }
        """
        real_label = np.array([self.id2name[x] for x in self.label])
        real_valid_label = np.array([self.id2name[x] for x in self.valid_label])

        data_dict = {
            'train_data': self.data,
            'train_label': real_label,
            'valid_data': self.valid,
            'valid_label': real_valid_label,
            'pruning method': self.pruning,
            'id2name': self.id2name
        }

        return data_dict


class C4_5(DecisionTree):
    def __init__(self, df, attrs2discretize=[], max_depth=0, valid_rate=0., valid_ix=[], pruning='none', random_state=42):
        """
        Initialize C4_5, data will be shuffled.

        Parameters
        ----------
        df : DataFrame
            the training data
        attrs2discretize : list
            attributes to be discretized
        max_depth : int
            maximum depth of the tree
        valid_rate : float
            rate of validation data
        valid_ix : list
            indices of validation data
        pruning : str
            way of pruning
        random_state : int
            random seed
        """
        if len(attrs2discretize) != 0:
            new_df = discretize(df=df, attrs=attrs2discretize)

        self.df_after_dis = new_df.copy()
        data, label, attr_dict, id2name = get_datas(new_df)
        train_ix = np.setdiff1d(np.arange(data.shape[0]), valid_ix)

        # shuffle
        np.random.seed(random_state)
        shuffle_ix = np.random.permutation(len(data))
        data = data[shuffle_ix]
        label = label[shuffle_ix]

        if valid_rate != 0:
            SPLIT = int(data.shape[0] * valid_rate)
            self.valid = data[:-SPLIT]
            self.valid_label = label[:-SPLIT]
            self.data = data[:SPLIT]
            self.label = label[:SPLIT]
        else:
            self.valid = np.array([data[ix] for ix in valid_ix])
            self.valid_label = np.array([label[ix] for ix in valid_ix])
            self.data = np.array([data[ix] for ix in train_ix])
            self.label = np.array([label[ix] for ix in train_ix])

        self.id2name = id2name
        self.pruning = pruning

        super().__init__(
            data=self.data,
            label=self.label,
            attr_dict=attr_dict,
            valid=self.valid,
            valid_label=self.valid_label,
            pruning=self.pruning,
            id2name=self.id2name)

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

        IV = 0
        for val in attr_val:
            label_temp = label[data[:, attr] == val]
            if len(label_temp) == 0:
                continue
            IV += (len(label_temp) / len(data)) * np.log2(len(label_temp) / len(data))

        return gain

    def datas(self):
        """
        Return a dictionary containing data and labels in the following format:
        {
            'df after discretizing': DataFrame,
            'train_data': 2d array,
            'train_label': 1d array,
            'valid_data': 2d array,
            'valid_label': 1d array,
            'pruning method': str,
            'id2name': dict
        }
        """

        real_label = np.array([self.id2name[x] for x in self.label])
        real_valid_label = np.array([self.id2name[x] for x in self.valid_label])

        data_dict = {
            'df after discretizing': self.df_after_dis,
            'train_data': self.data,
            'train_label': real_label,
            'valid_data': self.valid,
            'valid_label': real_valid_label,
            'pruning method': self.pruning,
            'id2name': self.id2name
        }

        return data_dict


class CART(DecisionTree):
    def __init__(self, df, attrs2discretize=[], max_depth=0, valid_rate=0., valid_ix=[], pruning='none', random_state=42):
        if len(attrs2discretize) != 0:
            new_df = discretize(df=df, attrs=attrs2discretize)

        self.df_after_dis = new_df.copy()
        data, label, attr_dict, id2name = get_datas(new_df)
        train_ix = np.setdiff1d(np.arange(data.shape[0]), valid_ix)

        # shuffle
        np.random.seed(random_state)
        shuffle_ix = np.random.permutation(len(data))
        data = data[shuffle_ix]
        label = label[shuffle_ix]

        if valid_rate != 0:
            SPLIT = int(data.shape[0] * valid_rate)
            self.valid = data[:-SPLIT]
            self.valid_label = label[:-SPLIT]
            self.data = data[:SPLIT]
            self.label = label[:SPLIT]
        else:
            self.valid = np.array([data[ix] for ix in valid_ix])
            self.valid_label = np.array([label[ix] for ix in valid_ix])
            self.data = np.array([data[ix] for ix in train_ix])
            self.label = np.array([label[ix] for ix in train_ix])

        self.id2name = id2name
        self.pruning = pruning

        super().__init__(
            data=self.data,
            label=self.label,
            attr_dict=attr_dict,
            valid=self.valid,
            valid_label=self.valid_label,
            pruning=self.pruning,
            id2name=self.id2name)
