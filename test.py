import cv2
import numpy as np
from keras.models import load_model

# Carregar o modelo treinado
model = load_model('models/yam.h5')

# Carregar uma nova imagem para teste
imagem_teste = cv2.imread('inhame.jpg')

# Redimensionar a imagem para o tamanho esperado pelo modelo (224x224 neste caso)
imagem_teste_redimensionada = cv2.resize(imagem_teste, (224, 224))

imagem_teste_normalizada = imagem_teste_redimensionada / 255.0

imagem_teste_processada = np.expand_dims(imagem_teste_normalizada, axis=0)

# Fazer a previsão com o modelo
previsao = model.predict(imagem_teste_processada)

# Transformar as probabilidades em porcentagem
probabilidades_percentuais = previsao * 100

# Exibir as probabilidades em formato de porcentagem para cada classe
print(f'Probabilidades para cada classe: {probabilidades_percentuais}')

# A classe com maior probabilidade será a classe predita
classe_predita = np.argmax(previsao)

print(f'A classe predita para a imagem é: {classe_predita}')
