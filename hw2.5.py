import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


print('-----------------------------------------------------------------------------------')
print('Задание 1')

x = np.array([1, 5, 10, 15, 20])
y1 = np.array([1, 7, 3, 5, 11])
y2 = np.array([4, 3, 1, 8, 12])

plt.plot(x, y1, '-ro', label='line 1')
plt.plot(x, y2, '-.go', label='line 1')
plt.legend()

plt.show()


print('-----------------------------------------------------------------------------------')
print('Задание 2')

grid = plt.GridSpec(2, 2)
ax = plt.subplot(grid[0, :])
x = np.arange(1, 6)
y = np.array([1, 7, 6, 3, 5])
ax.plot(x, y)

ax = plt.subplot(grid[1, 0])
y = np.array([9, 4, 2, 4, 9])
ax.plot(x, y)

ax = plt.subplot(grid[1, 1])
y = np.array([-7, -4, 2, -4, -7])
ax.plot(x, y)

plt.show()


print('-----------------------------------------------------------------------------------')
print('Задание 3')

x = np.arange(-5, 6)
y = x ** 2
plt.plot(x, y)
plt.annotate('min', xy=(0, 0), xytext=(0, 10),
             arrowprops=dict(facecolor='green'))

plt.show()


print('-----------------------------------------------------------------------------------')
print('Задание 4')

x = np.random.uniform(0, 7, 700)
y = np.random.normal(0, 7, 700)

fig, ax = plt.subplots()
bins = np.arange(0, 8, 1)
h = ax.hist2d(x, y, bins=[bins, bins], cmap='viridis', vmin=0, vmax=10)

ax.set_xlim(0, 7)
ax.set_ylim(0, 7)
ax.set_xticks(np.arange(0, 8, 1))
ax.set_yticks(np.arange(0, 8, 1))
fig.colorbar(h[3], ax=ax, shrink=0.5, anchor=(0, 0), aspect=5)

plt.show()


print('-----------------------------------------------------------------------------------')
print('Задание 5')

x = np.linspace(0, 5, 100)
y = np.cos(np.pi * x)

plt.plot(x, y, 'r')
plt.fill_between(x, y, 0, color='blue', alpha=0.6)

plt.show()


print('-----------------------------------------------------------------------------------')
print('Задание 6')

x = np.linspace(0, 5, 1000)
y = np.cos(np.pi * x)

mask = y < -0.5
y[mask] = np.nan

plt.plot(x, y, linewidth=4.0)
plt.ylim(-1, 1)

plt.show()


print('-----------------------------------------------------------------------------------')
print('Задание 7')

x = np.arange(7)
y = np.arange(7)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
styles = ['pre', 'mid', 'post']

for ax, style in zip(axes, styles):
    ax.step(x, y, where=style, color='green', marker='o')
    ax.grid(True)
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(7))
    ax.axis('equal')

plt.tight_layout()
plt.show()


print('-----------------------------------------------------------------------------------')
print('Задание 8')

x = np.linspace(0, 10, 100)
y1 = -0.2 * (x ** 2) + 2 * x
y2 = -0.6 * (x ** 2) + 6 * x
y3 = -4.0 / 7 * (x ** 2) + 8 * x

plt.fill_between(x, y1, 0, color='blue', label='y1')
plt.fill_between(x, y2, y1, color='orange', label='y2')
plt.fill_between(x, y3, y2, color='green', label='y3')
plt.ylim(0, 29)
plt.legend()

plt.show()


print('-----------------------------------------------------------------------------------')
print('Задание 9')

values = [15, 20, 25, 10, 30]
labels = ['BMW', 'Toyota', 'Ford', 'AUDI', 'Jaguar']

explode = [0] * len(labels)
explode[labels.index('BMW')] = 0.2

plt.figure()
plt.pie(
    values,
    labels=labels,
    explode=explode
)

plt.tight_layout()
plt.show()


print('-----------------------------------------------------------------------------------')
print('Задание 10')

plt.figure()
plt.pie(
    values, 
    labels=labels, 
    wedgeprops={'width': 0.5}
)

plt.tight_layout()
plt.show()