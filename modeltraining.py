# Google Drive ile baglanti kurulmasi

from google.colab import drive
drive.mount('/content/drive')

# Kutuphanelerin kurulmasi

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Veri artirma ile dataset hazirligi

# Parametreler

img_height = 224
img_width = 224
batch_size = 16

dataset_path = "/content/drive/MyDrive/dataset"

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',  # 4 sınıf için sparse
    subset='training',
    seed=42
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation',
    seed=42
)

# Sınıf etiketlerinin yazdırılması

print("Sınıf etiketleri:", train_generator.class_indices)

# Modelin olusturulmasi - MobileNetV2

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet')

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 sınıf için softmax
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model egitimi

epochs = 10

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# Modelin kaydedilmesi

model.save("model.h5")

# TFLite formatına donusturme islemi

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

# Etiket dosyasinin olusturulmasi

with open("labels.txt", "w") as f:
    f.write("ayasofya\ndiger\ngalata_kulesi\nkiz_kulesi")

from google.colab import files
from tensorflow.keras.preprocessing import image
import numpy as np

# Gorsel girdisi ile modelin test edilmesi

uploaded = files.upload()
img_path = list(uploaded.keys())[0]

# On isleme

img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Tahmin

prediction = model.predict(img_array)

# Etiketlerin yüklenmesi

labels = {v: k for k, v in train_generator.class_indices.items()}
predicted_index = np.argmax(prediction[0])
predicted_label = labels[predicted_index]
confidence = prediction[0][predicted_index]

# Tahmin ciktisi

print(f"✅ Tahmin: {predicted_label} (%{confidence * 100:.2f} olasılıkla)")