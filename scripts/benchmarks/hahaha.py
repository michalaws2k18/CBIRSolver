import pandas as pd
import numpy as np
data = {'InceptionResNetV2': {'precison': 0.26666666666666666, 'recall': 0.04, 'TP': 4, 'FP': 11},
        'histogram': {'precison': 0.06666666666666667, 'recall': 0.01, 'TP': 1, 'FP': 14},
        ' hist_norm': {'precison': 0.06666666666666667, 'recall': 0.01, 'TP': 1, 'FP': 14},
        'hist_gray': {'precison': 0.13333333333333333, 'recall': 0.02, 'TP': 2, 'FP': 13},
        'hist_gray_norm': {'precison': 0.06666666666666667, 'recall': 0.01, 'TP': 1, 'FP': 14},
        'hist_gray_equal': {'precison': 0.06666666666666667, 'recall': 0.01, 'TP': 1, 'FP': 14},
        'hist_equal': {'precison': 0.3333333333333333, 'recall': 0.0125, 'TP': 5, 'FP': 10},
        'hist_gray_equal_clahe': {'precison': 0.06666666666666667, 'recall': 0.01, 'TP': 1, 'FP': 14},
        'hist_equal_clahe': {'precison': 0.06666666666666667, 'recall': 0.01, 'TP': 1, 'FP': 14},
        'CCV': {'precison': 0.13333333333333333, 'recall': 0.02, 'TP': 2, 'FP': 13},
        'CCV_hist': {'precison': 0.13333333333333333, 'recall': 0.02, 'TP': 2, 'FP': 13}}


df = pd.DataFrame(data)
print(df.head())
df.loc['n_of_res'] = 16
df.loc['image_path'] = 'some_pathxd'
print(df)
print(type(data['InceptionResNetV2'].keys()))
npa = np.array(data)
print(len(npa.flatten()))
print(len(npa.flatten().flatten()))
