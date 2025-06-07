import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

iris = sns.load_dataset('iris')

target = []
for ind, row in iris.iterrows():
    species = row['species']
    if species == 'setosa':
        target.append(0)
    elif species == 'versicolor':
        target.append(1)
    else:
        target.append(2)
iris['target'] = target
iris = iris.drop(columns=['species'])

filtered_data = iris[(iris['target'] == 1) | (iris['target'] == 2)]
X = filtered_data[['sepal_length', 'sepal_width', 'petal_width']].values
y = filtered_data['target'].values

class_1 = X[y == 1]
class_2 = X[y == 2]

pca_1 = PCA(n_components=3)
pca_1.fit(class_1)

print(pca_1.components_)
print(pca_1.explained_variance_)
print(pca_1.mean_)

pca_2 = PCA(n_components=3)
pca_2.fit(class_2)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

ax.scatter(class_1[:, 0], class_1[:, 1], class_1[:, 2], c='blue', label='versicolor')
ax.scatter(class_2[:, 0], class_2[:, 1], class_2[:, 2], c='red', label='virginica')

def plot_vectors(ax, pca, label):
    start = pca.mean_
    for i, (component, variance) in enumerate(zip(pca.components_, pca.explained_variance_)):
        scale = np.sqrt(variance)
        end = start + component * scale
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            label=f'{label} dir {i + 1}',
        )


plot_vectors(ax, pca_1, label='versicolor')
plot_vectors(ax, pca_2, label='virginica')

ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
ax.set_zlabel('petal_width')
ax.legend()

plt.show()