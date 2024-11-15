import pytest
from app import app
from unittest.mock import patch

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

@patch('app.image_processor.process_image')
@patch('app.model.predict')
def test_predict_valid_image(mock_predict, mock_process_image, client):
    # Mockando as funções dependentes
    mock_process_image.return_value = [[[[0.1, 0.2, 0.3]]]]
    mock_predict.return_value = [[0.1, 0.8, 0.1]]

    response = client.post('/predict', json={'imagem': 'http://example.com/image.jpg'})
    data = response.get_json()

    assert response.status_code == 200
    assert data['classe'] == 'Ora-pro-nobis'
    assert 'acuracia' in data

@patch('app.image_processor.process_image')
def test_predict_invalid_image(mock_process_image, client):
    mock_process_image.return_value = {'error': 'Failed to download image'}

    response = client.post('/predict', json={'imagem': 'http://example.com/image.jpg'})
    data = response.get_json()

    assert response.status_code == 400
    assert data['error'] == 'Failed to download image'

def test_predict_exception_handling():
    with patch('services.image_processor.process_image') as mock_process_image:
        # Mockar a função para lançar uma exceção
        mock_process_image.side_effect = Exception("Unexpected error")

        # Criar cliente de teste do Flask
        client = app.test_client()

        # Fazer uma requisição POST com dados fictícios
        response = client.post('/predict', json={'imagem': 'http://example.com/fake.jpg'})

        # Verificar o status code e a resposta JSON
        assert response.status_code == 500
        assert response.json == {'error': 'Internal server error'}