import numpy as np


# Суммирование в массиве
# rng = np.random.default_rng(1)
# s = rng.random(50)

# print(s)
# print(sum(s))
# print(np.sum(s))

# a = np.array([
#     [1, 2, 3, 4, 5],
#     [6, 7, 8, 9, 10]
# ])
# print(np.sum(a))
# print(np.sum(a, axis=0))
# print(np.sum(a, axis=1))

# print(np.min(a))
# print(np.min(a, axis=0))
# print(np.min(a, axis=1))

# print(a.min())
# print(a.min(0))
# print(a.min(1))

# ---------------------

# # Nan - not a number
# print(np.namin(a))
# print(np.namin(a, axis=0))
# print(np.namin(a, axis=1))

# ---------------------

# Транслирование - broadcasting
# Набор правил, которые позволяют осуществлять бинарные операции
# с массивами разных форм и размеров

# a = np.array([0, 1, 2])
# b = np.array([5, 5, 5])
# print(a + b)
# print(a + 5)
# # 5 => [5, 5, 5]
# a = np.array([[0, 1, 2], [3, 4, 5]])
# print(a + 5)

# a = np.array([0, 1, 2])
# b = np.array([[0], [1], [2]])
# print(a + b)

# ---------------------

# Правила
# 1. Если размерности массивов отличаются, то форма массива с меньшей размерностью
# дополняется 1 с левой стороны
# 2. Если формы массивов не совпадают в каком-то измерении, то если у массива форма равна 1, то он растягивается до соответствия
# формы второго массива
# 3. Если после применения этих правил в каком-либо измерении размеры отличаются и ни один из них не равен 1, то мы не можем сделать транслирование и сложить массивы. Генерируется ошибка

# a = np.array( [[0, 1, 2], [3, 4, 5]] )
# b = np.array([5])

# print(a + b)
# print(a.ndim, a.shape)
# print(b.ndim, b.shape)

# a         (2, 3)
# b (1,) => (1, 1) -> (2, 3)

# a = np.ones((2, 3))
# b = np.arange(3)
# print(a)
# print(b)

# print(a.ndim, a.shape)
# print(b.ndim, b.shape)

# # (2, 3) -> (2, 3) -> (2, 3)
# # (3,)   -> (1, 3) -> (2, 3)
# c = a + b
# print(c, c.shape)

# a = np.arange(3).reshape((3, 1))
# b = np.arange(3)

# print(a)
# print(b)
# print(a.ndim, a.shape)
# print(b.ndim, b.shape)

# (3, 1) -> (3, 1) -> (3, 3)
# (3,)   -> (1, 3) -> (3, 3)

# c = a + b
# print(c, c.shape)

# a = np.ones((3, 2))
# b = np.arange(3)

# 2 (3, 2) -> (3, 2) -> (3, 2)
# 1 (3,)   -> (1, 3) -> (3, 3)

# ---------------------

## Q1. Что изменить, чтобы устранить ошибку?

# X = np.array([
#     [1, 2, 3, 4, 5, 6, 7, 8, 9],
#     [9, 8, 7, 6, 5, 4, 3, 2, 1]
# ])
# Xmean = X.mean(0)
# print(Xmean)

# Xcenter = X - Xmean
# print(Xcenter)
# Xmean1 = X.mean(1)
# print(Xmean1)
# Xmean1 = Xmean1[:, np.newaxis]
# Xcenter1 = X - Xmean1
# print(Xcenter1)


# x = np.linspace(0, 5, 50)
# y = np.linspace(0, 5, 50)[:, np.newaxis]
# z = np.sin(x) ** 3 + np.cos(20 + y * x) * np.sin(y)
# print(z.shape)

# ---------------------

# import matplotlib.pyplot as plt

# plt.imshow(z)
# plt.colorbar()
# plt.show()

# ---------------------

# Сравнение - определить, какие элементы массива удовлетворяют условию
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
# print(x < 3)
# print(np.less(x, 3))

# print(np.sum(x < 3)) # количество элементов

# print(np.sum(y < 4, axis=0))
# print(np.sum(y < 4, axis=1))
# print(np.sum(y < 4))

# ---------------------

# # & | ^ -

## 2. Пример для y. Вычислить количество элементов (по обоим размерностям), значения которых больше 3 и меньше 9

# # Маски - булевы массивы
# x = np.array([1, 2, 3, 4, 5])
# y = print(x < 3)

# print(x[x < 3])

# print(bin(42))
# print(bin(59))
# print(bin(42 & 59))

# ---------------------

# Векторизация индекса

# x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# index = [1, 5, 7]

# print(x[index])
# index = [[1, 5, 7], [2, 4, 8]]
# print(x[index])
# Форма результата соответствует индексу, а не исходному массиву

# x = np.arange(12).reshape(3, 4)
# print(x)
# print(x[2])

# можем уточнить
# print(x[2, [2, 0, 1]])
# print(x[1:, [2, 0, 1]])

# ---------------------

# С помощью векторизации можно изменять части массива

# x = np.arange(10)
# i = np.array([2, 1, 8, 4])
# print(x)
# x[i] = 999
# print(x)

# ---------------------

# Сортировка

# x = [3, 2, 3, 4, 2, 6, 7, 3, 6, 3, 2]
# print(sorted(x))
# print(np.sort(x))

# Структурированные массивы
# data = np.zeros(4, dtype={
#     'names': ('name', 'age'),
#     'formats': ('U10', 'i4')
# })
# print(data.dtype)
#
# name = ('name1', 'name2', 'name3', 'name4')
# age = [10, 20, 30, 40]
# data['name'] = name
# data['age'] = age
#
# print(data)
# print(data['age'] > 20)
# print(data[data['age'] > 20]['name'])

# Массивы записей
# data_rec = data.view(np.recarray)
# print(data_rec)
# print(data_rec[0])
# print(data_rec[-1].name)