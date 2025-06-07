import os

from keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

model = load_model('./lectures/data/model.h5')

img_path = './lectures/data/cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
# img = image.load_img()

import numpy as np

img_array = image.img_to_array(img)
print(img_array.shape)

img_batch = np.expand_dims(img_array, axis=0)
print(img_batch.shape)

# from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input

img_processed = preprocess_input(img_batch)

prediction = model.predict(img_processed)
print(prediction)

# from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions

# plt.imshow(img)
# plt.show()