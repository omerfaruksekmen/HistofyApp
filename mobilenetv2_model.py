# Google Drive ile baglanti kurulmasi

from google.colab import drive
drive.mount('/content/drive')

# Kutuphanelerin kurulmasi

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

# Veri artirma ile dataset hazirligi

# Parametreler

img_height = 224
img_width = 224
batch_size = 16

dataset_path = "/content/drive/MyDrive/dataset/train"

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
    class_mode='sparse',
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
    tf.keras.layers.Dense(4, activation='softmax')
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

# Test veri setinin yuklenmesi

mobilenet_test_datagen = ImageDataGenerator(rescale=1./255)

mobilenet_test_generator = mobilenet_test_datagen.flow_from_directory(
    '/content/drive/MyDrive/dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    shuffle=False
)

# Test veri seti ile modelin degerlendirilmesi

test_loss, test_accuracy = model.evaluate(mobilenet_test_generator)
print(f"MobileNetV2 Test Doğruluk Oranı: {test_accuracy:.4f}")

# Karışıklık matrisinin oluşturulması

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Gerçek ve tahmin edilen etiketler
y_true = mobilenet_test_generator.classes
y_pred_probs = model.predict(mobilenet_test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# Karışıklık Matrisi
cm = confusion_matrix(y_true, y_pred)
class_names = list(mobilenet_test_generator.class_indices.keys())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Karışıklık Matrisi (MobileNetV2)')
plt.show()

# Precision, Recall, F1-Score
print(classification_report(y_true, y_pred, target_names=class_names))

# Model özeti

model.summary()

# Model eğitimi, doğruluk ve kayıp oranları grafiği

# Eğitim geçmişi
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

# Grafik oluşturulması
plt.figure(figsize=(10, 4))

# 1. Grafik: Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Eğitim Doğruluğu', marker='o')
plt.plot(epochs_range, val_acc, label='Doğrulama Doğruluğu', marker='o')
plt.title('(MobileNetV2)\nModel Doğruluk Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid(True)

# 2. Grafik: Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Eğitim Kayıp', marker='o')
plt.plot(epochs_range, val_loss, label='Doğrulama Kayıp', marker='o')
plt.title('(MobileNetV2)\nModel Kayıp Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Kayıp (Loss)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Model katmanları grafiği

from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True, to_file='model.png')

# Modelin kaydedilmesi

model.save("mobilenetv2_model.h5")

# TFLite formatına donusturme islemi

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("mobilenetv2_model.tflite", "wb") as f:
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
