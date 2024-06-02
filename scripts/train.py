import os
from keras.api.applications import VGG16
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.api.models import Model
from keras.api.layers import Dense, Flatten
from keras.api.callbacks import ModelCheckpoint

# Definir o caminho para o conjunto de dados
train_data_dir = '../pre_processed_dataset/train'
validation_data_dir = '../pre_processed_dataset/validation'

# Definir o tamanho das imagens e o número de classes
img_width, img_height = 224, 224
num_classes = 3

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

# Carregar o modelo base pré-treinado
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Adicionar camadas de saída personalizadas
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Criar o modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Definir um callback para salvar o modelo periodicamente durante o treinamento
checkpoint = ModelCheckpoint('../models/pancs_checkpoint.keras', save_best_only=True)

# Treinar o modelo
model.fit(
    train_generator,
    epochs=20,
    callbacks=[checkpoint],
    validation_data=validation_generator
)

# Descongelar algumas camadas do modelo base para ajuste fino
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo novamente
model.fit(
    train_generator,
    epochs=20,
    callbacks=[checkpoint],
    validation_data=validation_generator
)

# Salvar o modelo treinado
model.save('../models/pancs.keras')
