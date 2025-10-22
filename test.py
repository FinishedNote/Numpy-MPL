import numpy as np

# slicing

array1 = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
,])

# print(array1[::-1, 0:1])
# print(array1[2,1])
# print(array1[1:3, 3:4])
# print(array1[1:, 2:])
# print(array1[::-1, 1])

# arithmetic

array2 = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

# print(array1 + array2)
# print(array1 - array2)
# print(array1 * array2)
# print(array1 / array2)
# print(array1 // array2)
# print(array1 % array2)
# print(array1 ** array2)

# broadcasting
"""
performance, avoid needless copies of data
rules: they are equal or one of them is 1
"""

array3 = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

array4 = np.array([
    [1], [2], [3], [4]
])

# print(array3.shape) --> (4, 4)
# print(array4.shape) --> (4, 1)

# print(array3 + array4)

array5 = np.array([2])
array6 = np.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11]
])

# print(array5.shape)
# print(array6.shape)

# print(array5 ** array6)

# aggregate functions

# print(array6.mean())
# print(np.mean(array6))
# print(np.sum(array6))
# print(np.max(array6))
# print(np.min(array6))
# print(np.argmin(array4)) --> index: 0
# print(np.argmax(array4)) --> index: 3

# print(np.sum(array2, axis=0)) --> column
# print(np.sum(array2, axis=1)) --> row

# filtering

ages = np.array([
    [21, 19, 17, 30, 75, 15],
    [26, 99, 14, 68, 42, 18]
])

# teenagers = ages[ages < 18]
# print(teenagers)

# adults = ages[(18 <= ages) & (65 > ages)]
# print(adults)

# adults = np.where((ages >= 18) & (ages < 65), ages, np.nan) --> last parameter is used to replace value, it can be anything 0, -1, np.nan, etc
# print(adults)

# random

rng = np.random.default_rng(seed=654) # --> seed is used to reproduce the same schema as mc and can be every number, but it's not required

# print(rng.integers(low=0, high=100)) --> just a number
# print(rng.integers(low=0, high=100, size=1)) --> 1 column
# print(rng.integers(low=0, high=100, size=2)) --> 2 columns
# print(rng.integers(low=0, high=100, size=(2, 3))) --> 2 rows 3 columns
# print(rng.uniform(size=4))
# print(rng.normal(size=4))
# print(rng.normal(size=4, scale=0.1))
# print(rng.exponential(size=4))
# print(rng.choice(ages[0]))
# rng.shuffle(ages[0])
# print(ages[0])