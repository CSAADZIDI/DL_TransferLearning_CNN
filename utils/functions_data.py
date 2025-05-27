import numpy as np
import os
import cv2

def generator_to_numpy(generator):
    """ convert generator to numpy 
    Args:
        generator (DataImageGenerator):

    Returns:
        X,y: numpy array
    """
    X = []
    y = []
    for i in range(len(generator)):
        X_batch, y_batch = generator[i]
        X.append(X_batch)
        y.append(y_batch)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y

def get_data(base_path: str, y_labels:list):
    """ load data from base_path

    Args:
        base_path (str): folder containing data
        y_labels (list) : list of labels

    Returns:
        X,y: numpy array
    """
    
    X = []  # liste pour stocker les images
    y = []  # liste pour stocker les étiquettes correspondantes

    # On parcourt les sous-dossiers du répertoire (un dossier par chiffre)
    for label in sorted(os.listdir(base_path)):
        # on ignore les fichiers qui ne sont pas des dossiers de chiffres
        if label not in y_labels:
            continue
        label_path = os.path.join(base_path, label)

        # On parcourt chaque image du dossier
        for file_name in os.listdir(label_path):
            file_path = os.path.join(label_path, file_name)
            # Lecture de l'image en niveaux de gris
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            # Convert to 3 channels
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # Now shape is (H, W, 3)
            # Resize to model input size (e.g., 224x224)
            resized_img = cv2.resize(img_rgb, (224, 224))
            
            if img is None:
                continue  # image illisible, on passe
            X.append(resized_img)           # on ajoute l'image à la liste
            y.append(1 if label == y_labels[0] else 0)    # on ajoute le label (1 if label == "PNEUMONIA" else 0)

    # Conversion des listes en tableaux NumPy
    X = np.array(X)
    y = np.array(y)
    return X, y

