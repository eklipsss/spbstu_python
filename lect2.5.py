# 1. Сценарий
# 2. Командная оболочка IPython
# 3. Jupyter

# 1
# plt.show() - запускается только один раз
# figure


import matplotlib.pyplot as plt
import numpy as np

# x = np.linspace(0, 10, 100)

# fig = plt.figure()
# plt.plot(x, np.sin(x))
# plt.plot(x, np.cos(x))
# plt.show()

# # IPython
# # %matplotlib
# # import matplotlib.pyplot as plt
# # plt.plot(...) - можно без show()
# # plt.draw()
# # чтобы скрыть консольную информацию, надо поставить ;

# # Jupyter
# # %matplotlib inline - в блокнот добавляется статическая картинка
# # %matplotlib notebook  - интерактивные графики

# fig.savefig('saved_images.png')

# print(fig.canvas.get_supported_filetypes())


# Два способа вывода графики
# - MATLAB-подобный стиль
# - ОО стиль
x = np.linspace(0, 10, 100)

# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(x, np.sin(x))

# plt.subplot(2, 1, 2)
# plt.plot(x, np.cos(x))

# fig, ax = plt.subplots(2)

# ax[0].plot(x, np.sin(x))
# ax[1].plot(x, np.cos(x))

# fig:plt.Figure - контейнер, который содержит объекты (СК, таксты, метки):
# ax: Axes - системы координат - прямоугольник, деления, метки

# Можем задавать цвета линий color
# - blue
# - rgbcmyk -> rg
# - 0.14 - градация серого от 0 до 1
# - RRGGBB: #FF00EE
# - RGB (1.0, 0.2, 0.3)
# - HTML - salmon

# Стиль линий
# - сплошная линия '-', solid
# - штриховая --, dashed
# - штрихпунктирная, dashdot
# - пунктирная :, dotted 

# fig = plt.figure()
# ax = plt.axes()

# ax.plot(x, np.sin(x), color='blue')
# ax.plot(x, np.sin(x - 1), color='g', linestyle='solid')
# ax.plot(x, np.sin(x - 2), color='0.75', linestyle='dashed')
# ax.plot(x, np.sin(x - 3), color='#FF00EE', linestyle='dashdot')
# ax.plot(x, np.sin(x - 4), color=(1.0, 0.2, 0.3),  linestyle='dotted')
# ax.plot(x, np.sin(x - 5), color='salmon')
# ax.plot(x, np.sin(x - 6), '--k')

# fig, ax = plt.subplots(4)

# ax[0].plot(x, np.sin(x))
# ax[1].plot(x, np.sin(x))
# ax[2].plot(x, np.sin(x))
# ax[3].plot(x, np.sin(x))

# ax[1].set_xlim(-2, 12)
# ax[1].set_ylim(-1.5, 1.5)

# ax[1].set_xlim(12, -2)
# ax[1].set_ylim(1.5, -1.5)

# ax[3].autoscale(tight=True)

# Подписи и легенда

# plt.subplot(3, 1, 1)
# plt.plot(x, np.sin(x))
# plt.title("Sine")
# plt.xlabel('x')
# plt.ylabel('sin(x)')

# plt.subplot(3, 1, 2)
# plt.plot(x, np.sin(x), '-g', label='sin(x)')
# plt.plot(x, np.cos(x), ':b', label='cos(x)')
# plt.title("Синус и косинус")
# plt.xlabel('x')
# plt.legend()

# plt.subplot(3, 1, 3)
# plt.plot(x, np.sin(x), '-g', label='sin(x)')
# plt.plot(x, np.cos(x), ':b', label='cos(x)')

# plt.title("Синус и косинус")
# plt.xlabel('x')
# plt.axis('equal')

# plt.legend()
# plt.subplots_adjust(hspace=0.5)

# x = np.linspace(0, 10, 30)
# plt.plot(x, np.sin(x), '--p', markersize=15, linewidth=4, markerfacecolor='white', markeredgecolor='gray', markeredgewidth=2)

# rng = np.random.default_rng(0)

# colors = rng.random(30)
# sizes = rng.random(30) * 30

# plt.scatter(x, np.sin(x), marker='o', c=colors, s=sizes)
# plt.colorbar()

# Если точек условно больше 1000, то plot предпочтительнее Из-за
# производительности - он быстрее

# x = np.linspace(0, 10, 50)
# dy = 0.4
# y = np.sin(x) + dy * np.random.randn(50)

# plt.errorbar(x, y, yerr=dy, fmt='.k')
# plt.fill_between(x, y - dy, y + dy, color='red', alpha=0.4)


def f(x, y):
    return np.sin(x) ** 5 + np.cos(20 + x * y) * np.cos(x)


x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)
X, Y = np.meshgrid(x, y)

Z = f(X, Y)

# plt.contour(X, Y, Z, cmap='jet')
# plt.contourf(X, Y, Z, cmap='RdGy')

c = plt.contour(X, Y, Z, color='red')
plt.clabel(c)
plt.imshow(Z, extent=[0, 5, 0, 5], cmap='RdGy', interpolation='gaussian', origin='lower')
plt.colorbar()

plt.show()