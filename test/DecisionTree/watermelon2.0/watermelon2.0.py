import hym.DecisionTree as dst
import numpy as np
import pandas as pd

# watermelon
attr_dict = {
    '色泽': ['青绿', '乌黑', '浅白'],
    '根蒂': ['蜷缩', '稍蜷', '硬挺'],
    '敲声': ['浊响', '沉闷', '清脆'],
    '纹理': ['清晰', '稍糊', '模糊'],
    '脐部': ['凹陷', '稍凹', '平坦'],
    '触感': ['硬滑', '软粘']
}

data = np.array([
    ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑'],
    ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑'],
    ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑'],
    ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑'],
    ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑'],
    ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘'],
    ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘'],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑'],
    ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑'],
    ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘'],
    ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑'],
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘'],
    ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑'],
    ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑'],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘'],
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑'],
    ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑']
])

# Labels for "好瓜" (0 = 否, 1 = 是)
labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

labels_str = np.where(labels == 1, '好瓜', '坏瓜')
df = pd.DataFrame(data, columns=attr_dict.keys())

df['label'] = labels_str
df.to_excel('output.xlsx', index=False)
# shuffle
# shuffle_ix = np.random.permutation(len(data))
# data = data[shuffle_ix]
# labels = labels[shuffle_ix]

train_ix = np.array([1, 2, 3, 6, 7, 10, 14, 15, 16, 17]) - 1
valid_ix = np.array([4, 5, 8, 9, 11, 12, 13]) - 1

for way in ['none', 'pre', 'post']:
    print(f'>> mine: {way}')
    # print(labels)
    tree = dst.ID3(df=df, valid_ix=valid_ix, pruning=way)
    # print(attr_dict)
    # print(train_ix)
    tree.fit()

    _, _, valid_data, valid_labels, _, _ = tree.datas()

    res = tree(valid_data)
    print(f'mine res: {res}')
    print(f'valid:    {valid_labels}')
    print('mine acc: ', np.mean(res == valid_labels))
    print()
    print('tree is: ')
    print(tree.tree)
    print()
    print()
