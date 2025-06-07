import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC


iris = sns.load_dataset('iris')
print(iris.head())

data = iris[["sepal_length", "petal_length", "species"]]
data_df = data[(data["species"] == "setosa") | (data["species"] == "versicolor")]

X = data_df[["sepal_length", "petal_length"]]
y = data_df["species"]

data_df_setosa = data_df[data_df["species"] == "setosa"]
data_df_versicolor = data_df[data_df["species"] == "versicolor"]

x1_p = np.linspace(min(data_df["sepal_length"]), max(data_df["sepal_length"]))
x2_p = np.linspace(min(data_df["petal_length"]), max(data_df["petal_length"]))

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=["sepal_length", "petal_length"]
)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))


ax[0].set_title('До удаления точек: ')
ax[0].scatter(data_df_setosa["sepal_length"], data_df_setosa["petal_length"])
ax[0].scatter(data_df_versicolor["sepal_length"], data_df_versicolor["petal_length"])

model = SVC(kernel='linear', C=10000)
model.fit(X, y)

ax[0].scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=200, facecolor='none', edgecolors='black')

y_p = model.predict(X_p)
y_p = np.where(y_p == 'setosa', 0, 1)
ax[0].contourf(X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.2, levels=2, cmap='rainbow', zorder=1)



ax[1].set_title('После удаления точек: ')
support_vector_indices = model.support_
non_support_vector_indices = [i for i in range(len(X)) if i not in support_vector_indices]
np.random.seed(42)
points_to_remove = np.random.choice(non_support_vector_indices, size=60, replace=False)

X = X.drop(index=points_to_remove).reset_index(drop=True)
y = y.drop(index=points_to_remove).reset_index(drop=True)
data_df = data_df.drop(index=points_to_remove).reset_index(drop=True)
# data_df_setosa.to_csv('./setosa.csv')
data_df_setosa = data_df[data_df["species"] == "setosa"]
data_df_versicolor = data_df[data_df["species"] == "versicolor"]
ax[1].scatter(data_df_setosa["sepal_length"], data_df_setosa["petal_length"])
ax[1].scatter(data_df_versicolor["sepal_length"], data_df_versicolor["petal_length"])

model = SVC(kernel='linear', C=10000)
model.fit(X, y)
ax[1].scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=200, facecolor='none', edgecolors='black')

y_p = model.predict(X_p)
y_p = np.where(y_p == 'setosa', 0, 1)
ax[1].contourf(X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.2, levels=2, cmap='rainbow', zorder=1)

print(model.support_vectors_)

plt.show()