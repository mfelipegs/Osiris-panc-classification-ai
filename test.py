import cv2
import numpy as np
from keras.api.models import load_model

# Mapeamento das classes
classes = {'hibiscus_rosa-sinensis': 0, 'ora-pro-nobis': 1, 'yam': 2}

# Carregar o modelo treinado
model = load_model('models/pancs.keras')

# Carregar uma nova imagem para teste
imagem_teste = cv2.imread('inhame2.jpg')

# Verificar se a imagem foi carregada corretamente
if imagem_teste is None:
    print('Erro ao carregar a imagem.')
else:
    # Redimensionar a imagem para o tamanho esperado pelo modelo (224x224)
    imagem_teste_redimensionada = cv2.resize(imagem_teste, (224, 224))

    # Normalizar a imagem
    imagem_teste_normalizada = imagem_teste_redimensionada / 255.0

    # Adicionar uma dimensão para corresponder ao formato esperado pelo modelo
    imagem_teste_processada = np.expand_dims(imagem_teste_normalizada, axis=0)

    # Fazer a previsão com o modelo
    previsao = model.predict(imagem_teste_processada)

    # Transformar as probabilidades em porcentagem
    probabilidades_percentuais = previsao * 100

    # Exibir as probabilidades em formato de porcentagem para cada classe
    print(f'Probabilidades para cada classe: {probabilidades_percentuais}')

    # A classe com maior probabilidade será a classe predita
    classe_predita = np.argmax(previsao)

    # Obter o nome da classe correspondente
    classe_predita_nome = list(classes.keys())[list(classes.values()).index(classe_predita)]

    print(f'A classe predita para a imagem é: {classe_predita_nome}')
