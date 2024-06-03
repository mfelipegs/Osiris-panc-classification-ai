#!/bin/bash

# Atualiza os pacotes e instala o Git LFS
sudo yum update -y
sudo yum install git-lfs -y

# Instala o Python 3 e suas dependências
sudo yum install python3 -y
sudo yum install python3-pip -y
pip3 install flask requests pillow

# Instala o Keras e o TensorFlow
echo "Instalando Keras..."
pip3 install keras==3.3
echo "Instalando TensorFlow..."
pip3 install tensorflow==2.16.1 --no-cache-dir

# Alocando memória do swap
echo "Alocando memória do swap..."
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo swapon --show
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Clona o repositório do Git com Git LFS
echo "Instalando o projeto..."
git lfs install
git clone https://github.com/mfelipegs/Osiris-panc-classification-ai.git
cd Osiris-panc-classification-ai

# Executa o projeto
echo "Executando o projeto..."
python3 app.py
