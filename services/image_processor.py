from PIL import Image
import numpy as np
import io
import requests
import logging

def process_image(image_url):
    try:
        # Definir um timeout para a requisição
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()  # Raises an exception for 4xx/5xx status codes

        image = Image.open(io.BytesIO(response.content))

        # Ensure image has 3 color channels (RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize image to expected size by the model (224x224)
        image_resized = image.resize((224, 224))

        # Convert image to numpy array
        image_array = np.array(image_resized)

        # Normalize image
        image_normalized = image_array / 255.0

        # Add a dimension to match the expected format by the model
        image_processed = np.expand_dims(image_normalized, axis=0)

        logging.info("Image processed successfully")
        return image_processed

    except requests.exceptions.RequestException as e:
        logging.error(f"Request error: {str(e)}")
        return {'error': 'Failed to download image'}

    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        return {'error': str(e)}
