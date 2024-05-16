from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint
import numpy as np

# Definir o caminho para o conjunto de dados
train_data_dir = 'pre_processed_dataset5/train'
validation_data_dir = 'pre_processed_dataset5/validation'

# Definir o tamanho das imagens e o número de classes
img_width, img_height = 224, 224
num_classes = 3

# Criar o modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Criar geradores de dados para treinamento e validação
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='sparse'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='sparse'
)

# Definir um callback para salvar o modelo periodicamente durante o treinamento
checkpoint = ModelCheckpoint('modelsv7/pancs_checkpoint.keras', save_best_only=True)

# Treinar o modelo com dados de validação
model.fit(
    train_generator,
    epochs=20,
    callbacks=[checkpoint],
    validation_data=validation_generator
)

# Salvar o modelo treinado
model.save('modelsv7/pancs.keras')
