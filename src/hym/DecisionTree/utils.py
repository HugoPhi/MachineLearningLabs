import pandas as pd
from copy import deepcopy
import numpy as np


def load_df(path):
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
    # get attribute dictionary
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
    # print(label)
    # print(id2name)
    # print(attr_dict)

    return data, label, attr_dict, id2name


def discretize(df, attrs):

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
        # print(mode)
        # print(label_temp)

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
        # new_df[attr] = new_df[attr].apply(lambda x: f'le{opt_split:5.2}' if x <= opt_split else f'gt{opt_split:5.2}')

    return new_df


# TODO: missing value
def handling_missing_value(self, df):
    return df
