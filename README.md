<p align="center">
   <img src="https://github.com/davitorress/Osiris-app/assets/104948713/5dfe90f9-43a4-442d-b499-04a74b9bfc0a" width="500">
</p>

<div align="center">
   
   ![GitHub language count](https://img.shields.io/github/languages/count/mfelipegs/Osiris-panc-classification-ai)
   ![GitHub last commit](https://img.shields.io/github/last-commit/mfelipegs/Osiris-panc-classification-ai)
   ![GitHub Release](https://img.shields.io/github/v/release/mfelipegs/Osiris-panc-classification-ai)

</div>

Osiris é um projeto de graduação em andamento que promove uma alimentação mais saudável por meio do uso de Plantas Alimentícias Não Convencionais (PANCs). Ele fornece informações de cultivo, diversas receitas e PANCs e permite que os usuários criem suas próprias receitas. Esta documentação descreve como utilizar a IA como API do Osiris.

Este projeto utiliza-se de uma API Flask como forma de testar (classificar) imagens de PANCs consultando um modelo de Rede Neural Convolucional desenvolvido com Keras e backend Tensorflow. Ao fazer uma requisição para o endpoint enviando o link da imagem da PANC, será retornado o resultado da classificação e sua acurácia. O projeto conta com uma imagem disponibilizada no Dockerhub, a API, o arquivo `.keras` do modelo e os scripts utilizados para a divisão do dataset em bases de treino e validação, bem como o script de treino utilizado para gerar o modelo.

---

&nbsp;

# Documentação da Osiris AI

**Tecnologias utilizadas:**

<p align="left">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
    <img src="https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white">
    <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white">
    <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white">
    <img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white">
</p>

---

## Como rodar a AI

<p align="left">
  <img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white">
</p>

- Se preferir, obtenha a imagem do projeto no Dockerhub [neste link](https://hub.docker.com/repository/docker/mfelipegs/osirisai-api/general) e rode em container.

#### Em modo desenvolvimento

- Utilize Git LFS para clonar o repositório. Após a clonagem:

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

&nbsp;

Certifique-se de que a porta 5000 está disponível para a aplicação.

## Endpoints

### Predict

### **POST** `/predict`

**Parameters**

- **body**

- `imagem`: Link de uma imagem a ser testada no modelo

```javascript
{
  "imagem": "https://res.cloudinary.com/..."
}
```

**Responses**

- **200** - Retorna a predição (classificação) da imagem

- `classe`: Classe predita da imagem (Hibiscus Rosa-Sinensis, Ora-pro-nobis ou Inhame)
- `acuracia`: Acurácia da predição em porcentagem (valor flutuante)

```javascript
{
  "classe": "string",
  "acuracia": "float"
}
```

- **400** - Requisição inconsistente

---
