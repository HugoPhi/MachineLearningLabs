import hym.DecisionTree as dt
import numpy as np

df = dt.load_df('./winequality_data.xlsx')
df = df.rename(columns={'quality label': 'label'})
column_names_without_last = df.columns[:-1].tolist()


opt = 0
opt_sed = 0

for _ in range(1):
    # sed = np.random.randint(0, 10000)
    sed = 5768
    # print(f'seed is {sed}')
    # sed = 7891
    for way in ['none']:
        # print(f">> way is: {way}")
        tree = dt.C4_5(
            df=df,
            attrs2discretize=column_names_without_last,
            valid_rate=0.5,
            random_state=sed,
            pruning=way)

        tree.fit()

        ddict = tree.datas()
        # print(ddict['df after discretizing'])

        res = tree(ddict['valid_data'])
        # print(f'mine res: {res}')
        # print(f'valid:    {ddict["valid_label"]}')
        print('mine acc: ', np.mean(res == ddict['valid_label']))
        print()
        print('tree is: ')
        print(tree.tree)
        print()
        print()

        if np.mean(res == ddict['valid_label']) > opt:
            opt = np.mean(res == ddict['valid_label'])
            opt_sed = sed

# print(f'optimal seed is {opt_sed}')
# print(f'opt acc is: {opt}')
