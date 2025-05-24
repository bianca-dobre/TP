import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from skimage.feature import local_binary_pattern

# Dictionar cu etichete
label_dict = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}

# Încarcă imaginile și etichetele
def load_dataset(folder_path):
    X = []
    y = []
    for emotion in os.listdir(folder_path):
        emotion_path = os.path.join(folder_path, emotion)
        if not os.path.isdir(emotion_path):
            continue
        label = label_dict.get(emotion.lower())
        if label is None:
            continue
        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (48, 48))
            X.append(img)
            y.append(label)
    return np.array(X), np.array(y)

# Extrage trăsături LBP
def extract_lbp_features(images):
    features = []
    for img in images:
        lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
        hist = hist.astype("float") / (hist.sum() + 1e-7)
        features.append(hist)
    return np.array(features)

# Încarcă datele
print("Loading training data...")
X_train_img, y_train = load_dataset("database/train")
print("Loading testing data...")
X_test_img, y_test = load_dataset("database/test")

# Extrage trăsături
print("Extracting features...")
X_train = extract_lbp_features(X_train_img)
X_test = extract_lbp_features(X_test_img)

# Antrenează modelul KNN
print("Training KNN classifier...")
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Testare
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
