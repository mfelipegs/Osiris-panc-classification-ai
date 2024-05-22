from flask import Flask, request, jsonify
from keras.api.models import load_model
import numpy as np
import logging
from services import image_processor

app = Flask(__name__)

# Configuração de logging
logging.basicConfig(level=logging.INFO)

# Mapeamento das classes
classes = {'Hibiscus rosa-sinensis': 0, 'Ora-pro-nobis': 1, 'Inhame': 2}

# Carregar o modelo treinado
model = load_model('modelsv16/pancs.keras')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receber a imagem da requisição POST
        data = request.get_json()
        logging.info(f"Received data: {data}")
        image_url = data['imagem']
        image_processed = image_processor.process_image(image_url)

        # Verificar se houve um erro ao processar a imagem
        if isinstance(image_processed, dict) and 'error' in image_processed:
            logging.error(f"Error processing image: {image_processed['error']}")
            return jsonify({'error': image_processed['error']}), 400

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

        logging.info(f"Prediction response: {response}")
        return jsonify(response)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
