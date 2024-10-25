import hym.DecisionTree as dt
import numpy as np


sed = np.random.randint(0, 10000)
print(f'seed is {sed}')
sed = 7891

df = dt.load_df('./iris.xlsx')
df = df.rename(columns={'species': 'label'})


for way in ['none']:
    print(f">> way is: {way}")
    tree = dt.C4_5(
        df=df,
        attrs2discretize=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
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
