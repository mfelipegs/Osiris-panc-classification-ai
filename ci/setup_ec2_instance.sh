#!/bin/bash
sudo yum update -y

sudo yum install python3 -y
sudo yum install python3-pip -y

pip3 install flask
pip3 install requests
pip3 install pillow

echo "Instalando Keras..."
pip3 install keras==3.3

echo "Instalando TensorFlow..."
pip3 install tensorflow==2.16.1 --no-cache-dir

echo "Alocando memória do swap..."
# Cria um arquivo de swap de 4GB
sudo fallocate -l 4G /swapfile

# Define as permissões corretas para o arquivo de swap
sudo chmod 600 /swapfile

# Inicializa o arquivo de swap
sudo mkswap /swapfile

# Ativa o arquivo de swap
sudo swapon /swapfile

# Verifica se o swap está ativo
sudo swapon --show

# Adiciona o arquivo de swap ao fstab para que ele seja ativado na inicialização
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

echo "Instalando o projeto..."
git clone https://github.com/mfelipegs/Osiris-panc-classification-ai.git
cd Osiris-panc-classification-ai

echo "Executando o projeto..."
python3 app.py