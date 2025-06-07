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


TRAIN_DATA_DIR = './lectures/data/train_data/'
VALIDATION_DATA_DIR = './lectures/data/val_data'
TRAIN_SAMPLES = 500
VALIDATION_SAMPLES = 500
NUM_CLASSES = 2
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 64

from tensorflow.keras.applications.mobilenet import preprocess_input, MobileNet

train_datagen = image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
)

val_datagen = image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=12345,
    class_mode='categorical',
)

val_generator = val_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=False,  
    class_mode='categorical',
)

from tensorflow.keras.layers import (
    Input,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.models import Model

def model_maker():
    base_model = MobileNet(include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    for layer in base_model.layers[:]:
        layer.trainable = False
    
    input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D()(custom_model)
    custom_model = Dense(64, activation='relu')(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    prediction = Dense(NUM_CLASSES, activation='softmax')(custom_model)
    return Model(inputs=input, outputs=prediction)

from tensorflow.keras.optimizers import Adam

model = model_maker()
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['acc']
)

import math
num_steps = math.ceil(float(TRAIN_SAMPLES) / BATCH_SIZE)

model.fit(
    train_generator,
    steps_per_epoch=num_steps,
    epochs=10,
    validation_data=val_generator,
    validation_steps=num_steps,
)
# Эпоха - полный шаг обучения. В рамках каждого просматривается весь набор
# данных. За шаг берем num_steps раз...

print(val_generator.class_indices)

model.save('./lectures/data/model.h5')


# img_path = './lectures/data/dog.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# # img = image.load_img()

# import numpy as np

# img_array = image.img_to_array(img)
# print(img_array.shape)

# img_batch = np.expand_dims(img_array, axis=0)
# print(img_batch.shape)

# from tensorflow.keras.applications.resnet50 import preprocess_input

# img_processed = preprocess_input(img_batch)

# from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions

# model = ResNet50()
# prediction = model.predict(img_processed)

# print(decode_predictions(prediction, top=5)[0])

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