import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

data = sns.load_dataset('iris')

target = []
for ind, row in data.iterrows():
    species = row['species']
    if species == 'setosa':
        target.append(0)
    elif species == 'versicolor':
        target.append(1)
    else:
        target.append(2)
data['target'] = target
data = data.drop(columns=['species'])

print(data.head())

filtered_data = data[(data['target'] == 1) | (data['target'] == 2)]
X = filtered_data[['sepal_length', 'sepal_width', 'petal_width']].values
y = filtered_data['target'].values

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

for i in range(n_clusters):
    cluster_points = X[kmeans.labels_ == i]
    ax.scatter(
        cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
        label=f'cluster {i + 1}'
    )

centers = kmeans.cluster_centers_
ax.scatter(
    centers[:, 0], centers[:, 1], centers[:, 2],
    c='m', s=100, marker='x', label='centers'
)

ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
ax.set_zlabel('petal_width')
ax.legend()

plt.show()