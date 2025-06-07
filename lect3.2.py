# Линейная регрессия

# Задача: на основе наблюдаемых точек построить прямую, которая отображает связь между двумя или более переменными.
# Регрессия пытается "подогнать" функцию к наблюдаемым данным, чтобы спрогнозировать новые данные.
# Линейная регрессия подгоняет данные к прямой линии, пытется установить линейную связь между переменными и предсказать новые данные.

import numpy as np
import random
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from numpy.linalg import inv, qr


# features, target = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=15, random_state=1)

# print(features.shape)
# print(target.shape)

# model = LinearRegression().fit(features, target)

# plt.scatter(features, target)

# x = np.linspace(features.min(), features.max(), 100)

# # y = kx + b
# plt.plot(x, model.coef_[0] * x + model.intercept_, color='red')

# plt.show()


# Простая линейная регрессия

# Линейная -> линейная зависимость.

# + прогнозирование на новых данных
# + анализ взаимного влияния переменных друг на друга

# - точки обучаемых данных НЕ будут точно лежать на прямой (шум) -> область погрешности
# - НЕ позволяет делать прогнозы ВНЕ диапазона имеющихся значений

# Данные, на основании которых разрабатывается модель - это выборка из совокупности, хотелось бы, чтобы это была РЕПРЕЗЕНТАТИВНАЯ выборка.

# Остатки (отклонения, ошибки) - расстояние между точками данных и ближайшими (по вертикали) точками на прямой

# y = w_0 + w_1 * x

# Задача состоит в минимизации остатков => разрыв между прямой и точками будет минимальным
# аусть R1... R10 остатки
# Просто сумму нельзя. Модуль нельзя, потому что не имеет производной в нуле. Чаще всего в качестве метрики используют сумму квадратов остатков.

# Обучение модели сводится к минимизации функции потерь.
# Аналитическое решение

# Решения: численное и аналитическое
# - численное - проще и доступнее, но требует большой вычислительной мощности, дает неточное решение
# - аналитическое - точное (математическая теория)

data = np.array([
    [1, 5],
    [2, 7],
    [3, 7],
    [4, 10],
    [5, 11],
    [6, 14],
    [7, 17],
    [8, 19],
    [9, 22],
    [10, 28],
])
x = data[:, 0]
y = data[:, 1]
n = len(x)
w_1 = (
    n * sum(x[i] * y[i] for i in range(n)) - sum(x[i] for i in range(n)) * sum(y[i] for i in range(n)) 
) / (n * sum(x[i] ** 2 for i in range(n))  - sum(x[i] for i in range(n)) ** 2  )
w_0 = sum(y[i] for i in range(n)) / n - w_1 * sum(x[i] for i in range(n)) / n

print(w_1, w_0)
# 2.4 -118.0

# 2. Метод обратных матриц
# y = w_1 * x + w_0; w = (w_0; w_1)
# X = [1 x_i]; Y = [y_i]

x_1 = np.vstack([x, np.ones(len(x))]).T
w = inv(x_1.transpose() @ x_1) @ (x_1.transpose() @ y)
print(w)


# 3. Разложение матриц
# X = Q * R -> w = R^(-1)Qy
# Минимизирует ошибку вычислений

Q, R = qr(x_1)
w = inv(R).dot(Q.T).dot(y)
print(w)


# 4. Градиентный спуск
# Это метод оптимизации, где используются производные и итерации.
# Частная производная (по одному из параметров) позволяет определить угловой коэффициент, и изменение параметра выполняется в ту сторону, где он макс/мин.
# Для больших угловых коэффициентов делается широкий шаг. Ширина шага вычисляется как доля от углового коэффициента -> скорость обучения.
# Чем выше скорость, тем быстрее будет работать система за счет снижения точности. Чем ниже скорость, тем больше времени займет обучение, но точность выше.

def f(x):
    return (x - 3) ** 2 + 4


def dx_f(x):
    return 2 * x - 6


# x = np.linspace(-10, 10, 100)
# ax = plt.gca()
# ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))

# # plt.plot(x, f(x))
# plt.plot(x, dx_f(x))

# plt.grid()

# plt.show()

L = 0.001
iterations = 100_000

x = random.randint(0, 5)

for i in range(iterations):
    d_x = dx_f(x)
    x -= L * d_x

print(x, f(x))

# Градиентный спуск работает с линейной регрессией.
# y = w_0 + w_1 * x

data = np.array([
    [1, 5],
    [2, 7],
    [3, 7],
    [4, 10],
    [5, 11],
    [6, 14],
    [7, 17],
    [8, 19],
    [9, 22],
    [10, 28],
])
x = data[:, 0]
y = data[:, 1]
n = len(x)

w1 = 0.0
w0 = 0.0

L = 0.001
iterations = 100_000
for i in range(iterations):
    D_w0 = 2 * sum(-y[i] + w0 + w1 * x[i] for i in range(n))
    D_w1 = 2 * sum(x[i] * (-y[i] + w0 + w1 * x[i]) for i in range(n))
    w1 -= L * D_w1
    w0 -= L * D_w0

print(w1, w0)

w1 = np.linspace(-10, 10, 100)
w0 = np.linspace(-10, 10, 100)


def E(w1, w0, x, y):
    return sum((y[i] - (w0 + w1 * x[i])) ** 2 for i in range(len(x)) )


W1, W0 = np.meshgrid(w1, w0)

EW = E(W1, W0, x, y)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(W1, W0, EW)

w1_fit = 2.4
w0_fit = 0.8

E_fit = E(w1_fit, w0_fit, x, y)

ax.scatter3D(w1_fit, w0_fit, E_fit, color='red')

plt.show()