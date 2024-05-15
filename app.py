from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
from keras.api.models import load_model

app = Flask(__name__)

# Mapeamento das classes
classes = {'hibiscus_rosa-sinensis': 0, 'ora-pro-nobis': 1, 'yam': 2}

# Carregar o modelo treinado
model = load_model('modelsv2/pancs.keras')

@app.route('/predict', methods=['POST'])
def predict():
    # Receber a imagem da requisição POST
    data = request.get_json()
    image_url = data['imagem']
    image = cv2.imdecode(np.asarray(bytearray(requests.get(image_url).content), dtype=np.uint8), cv2.IMREAD_COLOR)

    # Redimensionar a imagem para o tamanho esperado pelo modelo (224x224)
    image_resized = cv2.resize(image, (224, 224))

    # Normalizar a imagem
    image_normalized = image_resized / 255.0

    # Adicionar uma dimensão para corresponder ao formato esperado pelo modelo
    image_processed = np.expand_dims(image_normalized, axis=0)

    # Fazer a previsão com o modelo
    prediction = model.predict(image_processed)

    # Transformar as probabilidades em porcentagem
    probabilities_percent = prediction * 100

    # A classe com maior probabilidade será a classe predita
    predicted_class = np.argmax(prediction)

    # Obter o nome da classe correspondente
    predicted_class_name = list(classes.keys())[list(classes.values()).index(predicted_class)]

    # Preparar a resposta em formato JSON
    response = {
        'classe': predicted_class_name,
        'acuracia': float(probabilities_percent[0][predicted_class])
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
