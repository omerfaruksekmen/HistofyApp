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

dataset_path = "/content/drive/MyDrive/dataset/train"
img_height = 224
img_width = 224
batch_size = 16

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    shear_range=0.2,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Eğitim verisi
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training',
    seed=42
)

# Doğrulama verisi
val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation',
    seed=42
)

# Sınıf etiketleri
print("Sınıf etiketleri:", train_generator.class_indices)

# Modelin olusturulmasi - EfficientNetB0

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

# Son 100 katmanı eğitilebilir yap
base_model.trainable = True
for layer in base_model.layers[:-100]:
    layer.trainable = False

# 4 sınıflı model tanımı
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4, activation='softmax')  # ⬅️ 4 sınıf
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
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

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    shuffle=False
)

# Test veri seti ile modelin degerlendirilmesi

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Doğruluk Oranı: {test_accuracy:.4f}")

# Karışıklık matrisinin oluşturulması

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Gerçek ve tahmin değerleri
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# Karışıklık matrisi
cm = confusion_matrix(y_true, y_pred)
class_names = list(test_generator.class_indices.keys())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("Karışıklık Matrisi (EfficientNetB0)")
plt.show()

# Sınıf bazlı doğruluk, precision, recall
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

# Grafik oluştur
plt.figure(figsize=(10, 4))

# 1. Grafik: Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Eğitim Doğruluğu', marker='o')
plt.plot(epochs_range, val_acc, label='Doğrulama Doğruluğu', marker='o')
plt.title('(EfficientNetB0)\nModel Doğruluk Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid(True)

# 2. Grafik: Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Eğitim Kayıp', marker='o')
plt.plot(epochs_range, val_loss, label='Doğrulama Kayıp', marker='o')
plt.title('(EfficientNetB0)\nModel Kayıp Grafiği')
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

model.save("efficientnetb0_model.h5")

# TFLite formatına donusturme islemi

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("efficientnetb0_model.tflite", "wb") as f:
    f.write(tflite_model)

# Etiket dosyasinin olusturulmasi

with open("labels.txt", "w") as f:
    f.write("ayasofya\ndiger\ngalata_kulesi\nkiz_kulesi")

from google.colab import files
from tensorflow.keras.preprocessing import image

# Gorsel girdisi ile modelin test edilmesi

uploaded = files.upload()
img_path = list(uploaded.keys())[0]

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

# Normalize

processed_img = preprocess_input(img_batch)

# Tahmin

predictions = model.predict(processed_img)
predicted_index = np.argmax(predictions)
predicted_label = class_names[predicted_index]
confidence = predictions[0][predicted_index] * 100

# Sonucu yazdır

print(f"Tahmin: {predicted_label} ({confidence:.2f}%)")