import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.callbacks import ModelCheckpoint

# Criar o modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: ora-pro-nóbis, inhame, hibisco
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Definir um gerador de dados para treinamento
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'pre_processed_dataset/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse')  # Não é necessário especificar num_classes

# Definir um gerador de dados para validação
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
        'pre_processed_dataset/validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse')  # Não é necessário especificar num_classes

# Definir um callback para salvar o modelo periodicamente durante o treinamento
checkpoint = ModelCheckpoint('modelsv2/pancs_checkpoint.keras', save_best_only=True)

# Treinar o modelo com dados de validação
model.fit(train_generator, epochs=10, callbacks=[checkpoint], validation_data=validation_generator)

# Salvar o modelo treinado
model.save('modelsv2/pancs.keras')