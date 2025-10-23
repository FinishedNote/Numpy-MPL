import numpy as np

rdn_matrix = np.random.default_rng().uniform(low=0, high=100, size=(5, 5))
vector = np.arange(80, 85)

d = np.linalg.norm(rdn_matrix - vector, axis=1)
i = np.argmin(d)

r = rdn_matrix[i]

a = np.random.default_rng(seed=1).integers(low=0, high=50, size=(6, 4))
diff = a[:, np.newaxis, :] - a[np.newaxis, :, :]
dist = np.linalg.norm(diff, axis=1)

np.fill_diagonal(dist, np.inf)
i, j = np.unravel_index(dist.argmin(), dist.shape)

print(i, j)