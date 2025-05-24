import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from skimage.feature import local_binary_pattern
emotions = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}

def load_dataset(folder_path):
    X = [] #liste pour les images uploadees
    y = [] #liste pour les etiquettes des images
    for emotion in os.listdir(folder_path): #parcourir chaque directoire avec une emotion
        emotion_path = os.path.join(folder_path, emotion)
        if not os.path.isdir(emotion_path):
            continue
        label = emotions.get(emotion.lower()) #etiquette specifique
        if label is None:
            continue
        for image_name in os.listdir(emotion_path): #parcourir les images de directoire
            image_path = os.path.join(emotion_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, (48, 48))
            X.append(image) #ajouter l'image dans la liste
            y.append(label) #ajouter l'etiquette dans la liste
    return np.array(X), np.array(y)
def extract_lbp_features(images): #pour extraire les caracteristiques LBP des images
    features = []
    for image in images:
        lbp = local_binary_pattern(image, P=8, R=1, method='uniform') #P=voisins, R=rayon
        h, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9)) #l'histogramme de LBP avec un vector de 9 elements
        h = h.astype("float") / (h.sum() + 1e-7) #normaliser l'histogramme pour eviter la division par 0
        features.append(h)
    return np.array(features)
print("loading training data...")
X_train_image, y_train = load_dataset("database/train")
print("loading testing data...")
X_test_image, y_test = load_dataset("database/test")

print("extracting features using LBP...")
X_train = extract_lbp_features(X_train_image)
X_test = extract_lbp_features(X_test_image)

print("training KNN classifier...")
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("classification report:\n")
print(classification_report(y_test, y_pred))
