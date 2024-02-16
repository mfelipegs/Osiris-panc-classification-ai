import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# Carregar modelo
model = load_model('models/yam.h5')

# Definir gerador de dados para treinamento
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'pre_processed_dataset/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

# Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
model.fit(train_generator, epochs=10)

# Salvar o modelo treinado
model.save('models/yam.h5')
