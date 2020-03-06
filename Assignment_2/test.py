from scipy.stats import pearsonr,spearmanr
import numpy as np
a = [[1],[2],[3]]
b = [[1],[2],[3]]
a = np.array(a)
b = np.array(b)
# a = a.flatten()
# b = b.flatten()
# print(a)
# print(b)
# print(pearsonr(a,b))

a = np.reshape(a,[3])
print(a)