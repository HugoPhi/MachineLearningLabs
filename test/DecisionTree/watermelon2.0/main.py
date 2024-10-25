import hym.DecisionTree as dst
import numpy as np

df = dst.load_df('./watermelon2.0.xlsx')

train_ix = np.array([1, 2, 3, 6, 7, 10, 14, 15, 16, 17]) - 1
valid_ix = np.array([4, 5, 8, 9, 11, 12, 13]) - 1

for way in ['none', 'pre', 'post']:
    print(f'>> mine: {way}')
    tree = dst.ID3(df=df, valid_ix=valid_ix, pruning=way)
    tree.fit()

    valid_data = tree.datas()['valid_data']
    valid_labels = tree.datas()['valid_label']

    res = tree(valid_data)
    print(f'mine res: {res}')
    print(f'valid:    {valid_labels}')
    print('mine acc: ', np.mean(res == valid_labels))
    print()
    print('tree is: ')
    print(tree.tree)
    print()
    print()
