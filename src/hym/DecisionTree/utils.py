import pandas as pd
from copy import deepcopy
import numpy as np


def load_df(path):
    """
    Load a DataFrame from a specified file path.

    This function reads a CSV or Excel file from the given path and returns
    the data as a pandas DataFrame. If the file type is not supported, it
    prints an error message and exits the program.

    Parameters
    ----------
    path : str
        The file path to the data file. Supported file types are '.csv' and '.xlsx'.

    Returns
    -------
    pandas.DataFrame
        The loaded data as a DataFrame. If the file type is not supported, the
        function exits the program.
    """

    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.xlsx'):
        df = pd.read_excel(path)
    else:
        df = None
        print(f'error: {path}, filetype is not supported')
        exit(0)
    return df


def get_datas(df):
    """
    Extracts features, labels, attribute dictionary, and class name mapping from a DataFrame.

    This function processes a given DataFrame to generate a data array, a label array,
    an attribute dictionary mapping attribute names to their unique values, and a dictionary
    mapping class IDs to class names.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing features and a 'label' column.

    Returns
    -------
    tuple
        A tuple containing:
        - data : numpy.ndarray
            An array representation of the DataFrame.
        - label : numpy.ndarray
            An array of labels corresponding to the 'label' column in the DataFrame.
        - attr_dict : dict
            A dictionary where keys are attribute names and values are lists of unique
            attribute values, excluding the 'label' column.
        - id2name : dict
            A dictionary mapping class IDs to class names derived from the 'label' column.
    """

    attr_dict = {}
    for features in df.iloc[:]:
        unique_values = df[features].unique()
        attr_dict[features] = unique_values.tolist()

    # get label array & map class id to class name(e.g. 0 -> 'no', 1 -> 'yes')
    label, class_codes = pd.factorize(df['label'])
    id2name = {v: k for v, k in enumerate(class_codes)}
    attr_dict.pop('label')

    # get data array
    data = np.array(df.iloc[:])

    return data, label, attr_dict, id2name


def discretize(df, attrs):
    """
    Discretize the given DataFrame using the best-splitting method. Do not change the original DataFrame.

    The best-splitting method works by iterating over each attribute in the
    given list of attributes and computing the information gain of splitting
    the data at each possible value of the attribute. The attribute is then
    discretized by creating two new values for each possible split, i.e. one
    where the value is less than or equal to the split value and one where
    the value is greater than the split value.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing features and a 'label' column.
    attrs : list
        A list of attribute names to be discretized.

    Returns
    -------
    new_df : pandas.DataFrame
        The DataFrame with the given attributes discretized.
    """

    def Ent(label):
        prob = np.bincount(label) / len(label)
        res = np.array([p * np.log2(p) if p != 0 else 0 for p in prob])
        return -np.sum(res)

    new_df = deepcopy(df)
    nlabel, _ = pd.factorize(new_df['label'])
    for attr in attrs:
        arr = new_df[attr].to_numpy()
        ix = np.argsort(arr)
        arr = arr[ix]
        label_temp = nlabel[ix]

        mode = np.array([arr[0]] + [(arr[i] + arr[i + 1]) / 2 for i in range(len(arr) - 1)] + [arr[-1]])

        gain0 = Ent(label_temp)
        gains = []
        for m in mode:
            label_le = label_temp[arr <= m]
            label_gt = label_temp[arr > m]

            if len(label_le) == 0 or len(label_gt) == 0:
                gains.append(0)

            gain = gain0 - len(label_le) / len(nlabel) * Ent(label_le) - len(label_gt) / len(nlabel) * Ent(label_gt)
            gains.append(gain)

        ix = np.argmax(gains)
        opt_split = mode[ix]

        new_df[attr] = new_df[attr].apply(lambda x: f'≤{opt_split:.2f}' if x <= opt_split else f'>{opt_split:.2f}')

    return new_df


# TODO: missing value
def handling_missing_value(self, df):
    return df
