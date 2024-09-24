import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Paramètres du modèle
img_width, img_height = 224, 224

# Chemin vers le modèle pré-entraîné
model_path = 'expression_recognition_model.keras'

# Charger le modèle TensorFlow
model = tf.keras.models.load_model(model_path, compile=False)

app = Flask(__name__)



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



@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Convertir le FileStorage en BytesIO
        in_memory_file = BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)
        
        # Convertir l'image en JPG et la sauvegarder temporairement
        img = Image.open(in_memory_file)
        img = img.convert('RGB')  # Assurez-vous que l'image est en mode RGB
        img = img.rotate(-90, expand=True)
        img = img.crop((0, 200, img.width, 1100))
              
        temp_filename = 'temp_image.jpg'
        img.save(temp_filename, format='JPEG')
        
        try:
            # Chargement et préparation de l'image
            face_img = extract_and_resize_face(temp_filename, output_size=(img_width, img_height))
            if face_img is not None:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                img_array = img_to_array(face_img)
                img_array = np.expand_dims(img_array, axis=0)  # Crée un batch
                img_array /= 255.0  # Normalisation des pixels
                # Prédiction à partir de l'image
                prediction = model.predict(img_array)
                predicted_category = np.argmax(prediction)  # Retourne la catégorie prédite
            
                plt.imshow(face_img)
                plt.title(f'Prédit : {predicted_category}')
                plt.show()
                return jsonify({'predicted_category': int(predicted_category)})
            else:
                return jsonify({'predicted_category': int(-1)})
        finally:
            # Supprimer le fichier temporaire
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    