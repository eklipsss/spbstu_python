import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC


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

svm = SVC(kernel='linear')
svm.fit(X, y)

w = svm.coef_[0]
b = svm.intercept_[0]


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

class_1 = X[y == 1]
class_2 = X[y == 2]


ax.scatter(class_1[:, 0], class_1[:, 1], class_1[:, 2], c='blue', label='versicolor')
ax.scatter(class_2[:, 0], class_2[:, 1], class_2[:, 2], c='red', label='virginica')

x_range = (X[:, 0].min() - 1, X[:, 0].max() + 1)
y_range = (X[:, 1].min() - 1, X[:, 1].max() + 1)

xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], 50),
                     np.linspace(y_range[0], y_range[1], 50))
z = (-w[0] * xx - w[1] * yy - b) / w[2]
ax.plot_surface(
    xx, yy, z, alpha=0.3, color='gray',
    rstride=100, cstride=100
)

ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')
ax.set_zlabel('petal width')
ax.legend()

plt.show()