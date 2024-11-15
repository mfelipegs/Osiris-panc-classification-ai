import numpy as np
import requests
from services.image_processor import process_image
from unittest.mock import patch, MagicMock

def test_process_image_valid_url():
    with patch('services.image_processor.requests.get') as mock_get:
        # Mockar a resposta da requisição
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b'test_image_content'

        # Criar um mock para a imagem
        mock_image = MagicMock()
        mock_image.mode = 'RGB'
        mock_image.resize.return_value = mock_image
        mock_image.convert.return_value = mock_image

        # Configurar o array retornado pelo np.array
        fake_array = np.random.rand(224, 224, 3)  # Simula uma imagem de 224x224 com 3 canais
        with patch('PIL.Image.open', return_value=mock_image), \
             patch('numpy.array', return_value=fake_array):
            
            result = process_image('http://example.com/image.jpg')

            # Adicione uma dimensão ao array falso
            expected_shape = (1, 224, 224, 3)
            assert result.shape == expected_shape

def test_process_image_invalid_url():
    with patch('services.image_processor.requests.get') as mock_get:
        mock_get.side_effect = Exception("Failed to download image")
        result = process_image('http://example.com/invalid.jpg')
        assert result == {'error': 'Failed to download image'}

def test_process_image_non_rgb_image():
    with patch('services.image_processor.requests.get') as mock_get:
        # Mockar a resposta da requisição
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b'test_image_content'

        # Criar um mock para a imagem em formato não-RGB
        mock_image = MagicMock()
        mock_image.mode = 'L'  # Formato diferente de 'RGB'
        mock_image.convert.return_value = mock_image
        mock_image.resize.return_value = mock_image

        # Configurar o retorno para np.array
        mock_numpy_array = np.zeros((224, 224, 3))  # Simular o array de uma imagem RGB
        with patch('PIL.Image.open', return_value=mock_image), \
             patch('numpy.array', return_value=mock_numpy_array):
            result = process_image('http://example.com/image.jpg')

            # Validar que a imagem foi convertida e processada corretamente
            mock_image.convert.assert_called_once_with('RGB')
            assert result.shape == (1, 224, 224, 3)


def test_process_image_request_error():
    with patch('services.image_processor.requests.get') as mock_get:
        # Simular um erro de requisição (exemplo: timeout)
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        result = process_image('http://example.com/image.jpg')

        # Validar o retorno de erro
        assert result == {'error': 'Failed to download image'}