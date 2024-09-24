import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cv2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils import class_weight
from tensorflow.keras.applications import VGG16

#changer le working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


# Chemin vers le dossier contenant les données extraites
base_folder = 'Expression Recognition.v2i.coco'

# Paramètres
img_width, img_height = 224, 224
batch_size = 32
epochs = 10
num_classes = 3

# Mappage des catégories à des valeurs numériques
num_category_mapping = {'Neutral': 0, 'Yawn': 1, 'Sleep': 2}

# Fonction pour charger les annotations et les images depuis un fichier JSON
def load_annotations(folder_name):
    json_path = os.path.join(base_folder, folder_name, '_annotations.coco.json')
    with open(json_path) as file:
        coco_data = json.load(file)

    # Mappage des catégories à des noms
    category_mapping = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Création d'un dictionnaire pour associer chaque image à sa catégorie numérique
    image_label_mapping = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        category_name = category_mapping[ann['category_id']]
        if category_name in num_category_mapping:
            image_label_mapping[image_id] = num_category_mapping[category_name]
        else:
            image_label_mapping[image_id] = 0  # Assigner '0' si la catégorie n'est pas spécifiée

    # Création du tableau des résultats (nom de fichier et label)
    image_data = []
    for img in coco_data['images']:
        image_path = os.path.join(folder_name, img['file_name'])
        image_data.append((image_path, image_label_mapping.get(img['id'], 0)))  # '0' si aucune annotation
    
    return image_data


# Charger les données d'entraînement et de test
train_data = load_annotations('train')
test_data = load_annotations('test')
valid_data = load_annotations('valid')

# Fusionner les données d'entraînement et de test dans une seule liste
all_data = train_data + test_data + valid_data

# Affichage des premiers éléments pour vérifier
for data in all_data[:10]:  # Afficher les 10 premières associations
    print(data)


# Séparation des données en ensembles d'entraînement et de test
file_paths = [os.path.join(base_folder, data[0]) for data in all_data]
labels = [data[1] for data in all_data]

X_train, X_test, y_train, y_test = train_test_split(file_paths, labels, test_size=0.2, random_state=42)


def extract_and_resize_face(file_path, output_size=(img_width, img_height)):
    # Charger l'image en utilisant OpenCV
    image = cv2.imread(file_path)
    if image is None:
        return None  # Si l'image n'est pas trouvée ou ne peut pas être chargée

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Charger le classificateur en cascade pour les visages
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Extraire et redimensionner le premier visage trouvé
    for (x, y, w, h) in faces:
        face_image = image[y:y+h, x:x+w]
        face_image = cv2.resize(face_image, output_size)
        return face_image
    
    # Retourner None si aucun visage n'est trouvé
    return None

# Fonction pour charger et préparer les images
def process_images(file_paths, labels):
    images = []
    valid_labels = []  # Pour stocker les étiquettes correspondant aux images traitées
    for file_path, label in zip(file_paths, labels):
        face_img = extract_and_resize_face(file_path, output_size=(img_width, img_height))
        if face_img is not None:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            img_array = img_to_array(face_img)
            images.append(img_array)
            valid_labels.append(label)  # Ajouter l'étiquette correspondante
    if not images:
        return np.array([]), np.array([])  # Retourner des tableaux vides si aucun visage n'est détecté
    
    images = np.array(images)
    images /= 255.0  # Normalisation des pixels
    valid_labels = np.array(valid_labels)
    valid_labels = to_categorical(valid_labels, num_classes=num_classes)  # One-hot encoding des étiquettes
    return images, valid_labels



# Préparation des images
X_train, y_train = process_images(X_train, y_train)
X_test, y_test = process_images(X_test, y_test)

base_model = VGG16(include_top=False, input_shape=(img_width, img_height, 3))
# Construction du modèle
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(num_classes, activation='softmax')
])

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',  # La métrique à surveiller
    factor=0.5,              # Facteur de réduction du taux d'apprentissage, new_lr = lr * factor
    patience=3,              # Nombre d'époques sans amélioration après lesquelles le taux d'apprentissage sera réduit
    verbose=1,               # Activer la sortie détaillée
    min_lr=0.00001           # Valeur minimale que peut atteindre le taux d'apprentissage
)

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


y_train_labels = np.argmax(y_train, axis=1)
# Calcul des poids pour chaque classe
class_weights = class_weight.compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_train_labels), 
    y=y_train_labels
)

# Convertir en dictionnaire
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}


# Entraînement du modèle
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr],
    class_weight=class_weights_dict 
)

# Sauvegarde du modèle
model.save('expression_recognition_model.keras')


# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calcul de la matrice de confusion
cm = confusion_matrix(y_true, y_pred_classes)

# Affichage de la heatmap de la matrice de confusion
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=num_category_mapping.values(), yticklabels=num_category_mapping.values())
plt.title('Matrice de confusion')
plt.ylabel('Vraies classes')
plt.xlabel('Classes prédites')
plt.show()




#%%
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cv2

# Chemin vers le dossier contenant les données extraites
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Paramètres
img_width, img_height = 224, 224
batch_size = 32
epochs = 10
num_classes = 3

def extract_and_resize_face(file_path, output_size=(img_width, img_height)):
    # Charger l'image en utilisant OpenCV
    image = cv2.imread(file_path)
    if image is None:
        return None  # Si l'image n'est pas trouvée ou ne peut pas être chargée

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Charger le classificateur en cascade pour les visages
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Extraire et redimensionner le premier visage trouvé
    for (x, y, w, h) in faces:
        face_image = image[y:y+h, x:x+w]
        face_image = cv2.resize(face_image, output_size)
        return face_image
    
    # Retourner None si aucun visage n'est trouvé
    return None



# Fonction pour prédire une nouvelle image
def predict_image(file_path, model_path='expression_recognition_model.keras'):
    # Chargement du modèle
    model = tf.keras.models.load_model(model_path, compile=False)

    face_img = extract_and_resize_face(file_path, output_size=(img_width, img_height))
    if face_img is not None:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img_array = img_to_array(face_img)
        img_array = np.expand_dims(img_array, axis=0)  # Crée un batch
        img_array /= 255.0  # Normalisation des pixels
        # Prédiction à partir de l'image
        prediction = model.predict(img_array)
        predicted_category = np.argmax(prediction) 
        plt.imshow(face_img)
        plt.title(f'Prédit : {predicted_category}')
        plt.show()
        return predicted_category
    else:
        return -1





# Chemin vers le dossier contenant les images de validation
validation_folder = 'Image de validation'

# Récupération de tous les fichiers JPEG
image_files = [os.path.join(validation_folder, f) for f in os.listdir(validation_folder) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]

# Prédiction pour chaque image
predictions = {file: predict_image(file) for file in image_files}

# Calcul du nombre de lignes nécessaires en fonction du nombre d'images
num_images = len(image_files)
num_rows = (num_images + 3) // 4  # Nombre de lignes nécessaires pour 4 images par ligne

# Affichage des images et des prédictions
fig = plt.figure(figsize=(15, 3 * num_rows))  # Largeur fixe, hauteur ajustée en fonction du nombre de lignes
for i, file in enumerate(image_files, 1):
    category = predictions[file]
    img = load_img(file, target_size=(img_width, img_height))
    
    ax = fig.add_subplot(num_rows, 4, i)  # Configuration pour 4 images par ligne
    ax.imshow(img)
    ax.set_title(f'Predicted: {category}')
    ax.axis('off')  # Désactiver les axes pour une visualisation plus propre

plt.tight_layout()
plt.show()

# Affichage des résultats
for file, category in predictions.items():
    print(f'{file}: Predicted category = {category}')

