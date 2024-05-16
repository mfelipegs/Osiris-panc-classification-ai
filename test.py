import cv2
import numpy as np
from keras.models import load_model, Model
import matplotlib.pyplot as plt

# Mapeamento das classes
classes = {'hibiscus_rosa-sinensis': 0, 'ora-pro-nobis': 1, 'yam': 2}

# Carregar o modelo treinado
model = load_model('modelsv7/pancs.keras')

# Escolha uma camada intermediária para visualizar
layer_name = 'conv2d_1'

# Crie um novo modelo que retorna as ativações da camada escolhida
activation_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

# Carregar uma nova imagem para teste
imagem_teste = plt.imread('flor-de-hibisco-02.jpg')

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

    # Obtenha as ativações para a imagem de teste
    activations = activation_model.predict(imagem_teste_processada)

    # Visualize as ativações
    for i in range(activations.shape[-1]):
        activation_map = activations[0, :, :, i]
        activation_map = cv2.resize(activation_map, (224, 224))
        activation_map = (activation_map * 255).astype(np.uint8)
        cv2.imshow(f'Activation Map {i}', activation_map)

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

    cv2.waitKey(0)
    cv2.destroyAllWindows()
