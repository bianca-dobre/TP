import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
# Associer chaque emotion a une valeur numerique
label_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}

# lire les paires image-etiquette
def preparer_donnees(racine):
    donnees, classes = [], []
    noms = os.listdir(racine)

    for nom in noms:
        rep = os.path.join(racine, nom)
        if not os.path.isdir(rep) or nom.lower() not in label_map:
            continue

        for fichier in os.listdir(rep):
            chemin = os.path.join(rep, fichier)
            img = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                petite = cv2.resize(img, (48, 48))
                donnees.append(petite)
                classes.append(label_map[nom.lower()])

    return np.array(donnees), np.array(classes)

# Extraire une signature LBP pour chaque image
def creer_vecteurs_lbp(liste_images):
    resultats = []
    for i in liste_images:
        lbp = local_binary_pattern(i, P=8, R=1, method='uniform')
        vect, _ = np.histogram(lbp.ravel(), bins=np.arange(10), range=(0, 9))
        vect = vect.astype("float32")
        vect /= (vect.sum() + 1e-6)
        resultats.append(vect)
    return np.array(resultats)


def charger_base(dossier):
    return preparer_donnees(dossier)
    
def processus_complet():
    print("Debut du processus...")
    print("Lecture des donnees d'entrainement")
    images_appr, etiquettes_appr = charger_base("database/train")
    print("Lecture des donnees de test")
    images_test, etiquettes_test = charger_base("database/test")
    print("Calcul des LBP pour l'ensemble d'entrainement")
    X_train = creer_vecteurs_lbp(images_appr)
    print("Calcul des LBP pour l'ensemble de test")
    X_test = creer_vecteurs_lbp(images_test)
    print("Initialisation du classifieur KNN...")
    modele = KNeighborsClassifier(n_neighbors=3)
    modele.fit(X_train, etiquettes_appr)
    print("Prediction en cours...")
    sorties = modele.predict(X_test)
    print("Evaluation du modele :\n")
    print(classification_report(etiquettes_test, sorties))

if __name__ == "__main__":
    processus_complet()
