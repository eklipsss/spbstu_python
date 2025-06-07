# Нейронные сети:
# - сверточные (конволюционные) нейронные сети (CNN) - компьютерное зрение,
# классификация изображений
# - рекуррентные нейронные сети (RNN) - распознование рукописного текста,
# обработка естественного языка
# - генеративные состязательные сети (GAN) - создание художественных,
# музыкальных произведений
# - многослойный перцептрон - простейший тип НС

# Функция выпрямленных линейных единиц (ReLU)
# f(x) = x if x > 0 else 0

# 1. Начальные значения весов - случайные числа
# 2. Смешения = 0
# 3. Функция потерь

# Алгоритм обратного распространения ошибки
# - вычисление частных производных
# - градиентный спуск

# TensorFlow (Keras)
# PyTorch

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


img_path = './lectures/data/dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
# img = image.load_img()

import numpy as np

img_array = image.img_to_array(img)
print(img_array.shape)

img_batch = np.expand_dims(img_array, axis=0)
print(img_batch.shape)

from tensorflow.keras.applications.resnet50 import preprocess_input

img_processed = preprocess_input(img_batch)

from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions

model = ResNet50()
prediction = model.predict(img_processed)

print(decode_predictions(prediction, top=5)[0])

# Перенос обучения
# Сверточные слои: преобразуют признаки
# Полносвязные слои: классификация
# Начальные слои - слои обобщенным знанием, дальше - слои с
# узкоспециализированным знанием
# Каждый слой пропускает только те элементы, которые он может распознать
# Можно брать часть модели, добавить свои слои и получить новый выход

# Шаги
# 1. Организация данных - обучающие и проверочные данные + контрольные
# 2. Пайплайн подготовки
# 3. Аугментация данных - обогащение набора
# 4. Определение модели. Замораживаем коэффициенты слоев, которые не хотим менять
# Алгоритм оптимизатора, метрика оценки
# 5. Обучение модели -> итерации, пока метрика не станет приемлемой
# 6. Сохраняем модель и используем

# plt.imshow(img)
# plt.show()