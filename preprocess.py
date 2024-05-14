import cv2
import os
from sklearn.model_selection import train_test_split

def resize_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    if img is not None:
        h, w = img.shape[:2]
        if h > w:
            scale = target_size[0] / h
        else:
            scale = target_size[1] / w
        resized_img = cv2.resize(img, None, fx=scale, fy=scale)
        padded_img = cv2.copyMakeBorder(resized_img, 0, target_size[0] - resized_img.shape[0], 0, target_size[1] - resized_img.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded_img
    return None

base_dir = 'dataset/'
sets = ['train', 'validation']
classes = ['hibiscus_rosa-sinensis', 'ora-pro-nobis', 'yam']

for dataset in sets:
    for cls in classes:
        image_paths = [os.path.join(base_dir, dataset, cls, img) for img in os.listdir(os.path.join(base_dir, dataset, cls))]

        train_paths, test_paths = train_test_split(image_paths, test_size=0.1, random_state=42)
        train_paths, val_paths = train_test_split(train_paths, test_size=0.2, random_state=42)

        preprocessed_folder = f'pre_processed_dataset/{dataset}/{cls}'
        os.makedirs(preprocessed_folder, exist_ok=True)

        for img_path in train_paths:
            img = resize_image(img_path)
            if img is not None:
                img_normalized = img / 255.0
                img_name = os.path.basename(img_path)
                cv2.imwrite(os.path.join(preprocessed_folder, img_name), img_normalized * 255.0)

        for img_path in val_paths:
            img = resize_image(img_path)
            if img is not None:
                img_normalized = img / 255.0
                img_name = os.path.basename(img_path)
                cv2.imwrite(os.path.join(preprocessed_folder, img_name), img_normalized * 255.0)

        for img_path in test_paths:
            img = resize_image(img_path)
            if img is not None:
                img_normalized = img / 255.0
                img_name = os.path.basename(img_path)
                cv2.imwrite(os.path.join(preprocessed_folder, img_name), img_normalized * 255.0)
