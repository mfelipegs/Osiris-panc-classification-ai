#!/bin/bash

# Atualiza os pacotes e instala o Git e o Git LFS
sudo apt update
sudo apt install git -y
sudo apt install git-lfs -y

# Instala o Python 3 e venv
sudo apt install python3 -y
sudo apt install python3-venv -y

# Alocando memória do swap (opcional, apenas se necessário)
echo "Alocando memória do swap..."
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo swapon --show
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Cria e ativa um ambiente virtual Python
echo "Criando e ativando ambiente virtual Python..."
python3 -m venv myenv
source myenv/bin/activate

# Instala os pacotes no ambiente virtual
echo "Instalando pacotes..."
pip install flask requests pillow keras==3.3 tensorflow==2.16.1

# Clona o repositório do Git com Git LFS
echo "Instalando o projeto..."
git lfs install
git clone https://github.com/mfelipegs/Osiris-panc-classification-ai.git
cd Osiris-panc-classification-ai

# Executa o projeto
echo "Executando o projeto..."
nohup python3 app.py > output.log 2>&1 &
