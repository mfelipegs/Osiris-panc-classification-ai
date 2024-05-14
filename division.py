import os
import shutil
from sklearn.model_selection import train_test_split

# Diretório original com todas as imagens
dataset_dir = 'dataset'

# Diretório para as imagens de treinamento e validação
train_dir = 'dataset/train'
val_dir = 'dataset/validation'

# Criar diretórios para treinamento e validação se ainda não existirem
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Listar as classes de PANCs
classes = ['hibiscus_rosa-sinensis', 'ora-pro-nobis', 'yam']

# Dividir as imagens em conjuntos de treinamento e validação
for classe in classes:
    # Diretório original para a classe
    class_dir = os.path.join(dataset_dir, classe)

    # Listar todas as imagens para a classe
    images = os.listdir(class_dir)

    # Dividir as imagens em treino e validação (80% treino, 20% validação)
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    # Mover as imagens para os diretórios de treino e validação
    for image in train_images:
        src = os.path.join(class_dir, image)
        dst = os.path.join(train_dir, classe, image)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)

    for image in val_images:
        src = os.path.join(class_dir, image)
        dst = os.path.join(val_dir, classe, image)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
