import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

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

face = iio.imread("./images/koala.jpg")
gray_face = np.dot(face[...,:3], [0.2989, 0.5870, 0.1140])
plt.imshow(gray_face, cmap="gray")
# plt.axis("off")
# plt.show()
# print(gray_face.shape)
h = gray_face.shape[0]
w = gray_face.shape[1]
zoom = gray_face[h//4 : -h//4, w//4 : -w//4] # zoom x2 ~ 1/4
# zoom[zoom > 150] = 255
# zoom[zoom < 150] = 0
zoom = zoom[::2, ::2] # reduction in steps of 2
# plt.imshow(zoom, cmap="gray")
# plt.axis("off")
# plt.show()

# average deviation

np.random.seed(0)
A = np.random.randint(0, 100, [10, 5]).astype(float)
B = A.copy()
A = (A - A.mean(axis=0)) / A.std(axis=0) # or solution in b
# B = (B - np.average(B, axis=0)) / B.std(axis=0)
# print(A)
# print(B)

# mpl rnd exp

dataset = {f"{i}": np.random.randn(100).astype(float) for i in range(4)}
"""
def graph(data):
    fig, ax = plt.subplots(4, 1,sharex=True, sharey=True)
    for k, j in data.items():
        current_d = data[k] = j
        ax[int(k)].plot(current_d)
        ax[int(k)].set_title(f"experience {int(k)+1}")

    plt.savefig("./images/graph.png")
    plt.show()

graph(dataset)
"""

n = len(dataset)
def graph(data):
    fig, ax = plt.subplots(4, 1,sharex=True, sharey=True)
    for k, j in zip(data.keys(), range(0, n+1)):
        ax[int(k)].plot(data[k])
        ax[int(k)].set_title(f"experience {j+1}")

    plt.show()

graph(dataset)