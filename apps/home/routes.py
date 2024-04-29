# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound
from flask import  render_template, request, jsonify
import cv2
import numpy as np
import base64
from sklearn.svm import SVC
import numpy as np
from matplotlib import pyplot as plt
from skimage import feature
import joblib
@blueprint.route('/index')
@login_required
def index():

    return render_template('home/index.html', segment='index')


@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
@blueprint.route('/filter_image', methods=['POST'])
def filter_image():
    # Récupérer le fichier d'image à partir de la requête
    file = request.files['file']
    # Lire l'image
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    # Appliquer le filtre (exemple avec flou gaussien)
    filtered_img = cv2.GaussianBlur(img, (5, 5), 0)
    # Convertir l'image filtrée en base64 pour l'envoyer au frontend
    _, img_encoded = cv2.imencode('.jpg', filtered_img)
    filtered_img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return jsonify({'image': filtered_img_base64})
# Route pour la segmentation de l'image
@blueprint.route('/segment_image', methods=['POST'])
def segment_image():
    # Récupérer le fichier d'image à partir de la requête
    file = request.files['file']
    # Lire l'image
    cell_image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.COLOR_BGR2RGB)
    # Appliquer la segmentation (exemple avec seuillage)
    image_lab = cv2.cvtColor(cell_image, cv2.COLOR_BGR2LAB)
# Extraire la composante a* de l'image LAB
    _, a, _ = cv2.split(image_lab)

# Appliquer un seuillage d'Otsu
    _, binary_mask = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Remplir les trous dans le masque
    filled_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    filled_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

# Appliquer le masque sur l'image d'origine
    result_image = cv2.bitwise_and(cell_image, cell_image, mask=filled_mask)

# Convertir l'image en niveaux de gris
    gray_cell_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

# Trouver les contours dans l'image en niveaux de gris
    contours, _ = cv2.findContours(gray_cell_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Convertir l'image segmentée en base64 pour l'envoyer au frontend
    contour_image = cell_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    _, img_encoded = cv2.imencode('.png', result_image)
    segmented_img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return jsonify({'image': segmented_img_base64})

# Route pour la détection des contours de l'image
@blueprint.route('/contour_nimage', methods=['POST'])
def contour_imagre():
    # Récupérer le fichier d'image à partir de la requête
    file = request.files['file']
    # Lire l'image
    cell_image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    # Appliquer la segmentation (exemple avec seuillage)
    image_lab = cv2.cvtColor(cell_image, cv2.COLOR_BGR2LAB)
# Extraire la composante a* de l'image LAB
    _, a, _ = cv2.split(image_lab)

# Appliquer un seuillage d'Otsu
    _, binary_mask = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Remplir les trous dans le masque
    filled_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    filled_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

# Appliquer le masque sur l'image d'origine
    result_image = cv2.bitwise_and(cell_image, cell_image, mask=filled_mask)

# Convertir l'image en niveaux de gris
    gray_cell_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

# Trouver les contours dans l'image en niveaux de gris
    contours, _ = cv2.findContours(gray_cell_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Convertir l'image segmentée en base64 pour l'envoyer au frontend
    contour_image = cell_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    _, img_encoded = cv2.imencode('.jpg', contour_image)
    contours_img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return jsonify({'image': contours_img_base64})
@blueprint.route('/contour_image', methods=['POST'])
def contour_image():
    # Récupérer le fichier d'image à partir de la requête
    file = request.files['file']
    # Lire l'image
    cell_image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
# Fonction qui segmenter l'image
    def segmentation_cell_image(cell_image):
        image_lab = cv2.cvtColor(cell_image, cv2.COLOR_BGR2LAB)

# Extraire la composante a* de l'image LAB
        _, a, _ = cv2.split(image_lab)

# Appliquer un seuillage d'Otsu
        _, binary_mask = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Remplir les trous dans le masque
        filled_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        filled_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

# Appliquer le masque sur l'image d'origine
        result_image = cv2.bitwise_and(cell_image, cell_image, mask=filled_mask)

# Convertir l'image en niveaux de gris
        gray_cell_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

# Trouver les contours dans l'image en niveaux de gris
        contours, _ = cv2.findContours(gray_cell_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours , gray_cell_image,a,binary_mask, filled_mask ,result_image
# Définir une fonction pour extraire les caractéristiques morphologiques et texturales
    def extract_features(contour, gray_image):

       # Extraire les caractéristiques morphologiques
        area = cv2.contourArea(contour)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        enclosing_circle_center_x, enclosing_circle_center_y = (int(x), int(y))
        enclosing_circle_diameter = 2 * int(radius)
        perimeter = cv2.arcLength(contour, True)
        compactness = 4 * np.pi * area / perimeter**2 if perimeter>0 else 0
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        aspect_ratio = width / height if height>0 else 0

        # Extraire les caractéristiques de texture de Haralick
        x, y, w, h = cv2.boundingRect(contour)
        cell_roi = gray_image[y:y+h, x:x+w]
        glcm = feature.graycomatrix(cell_roi, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
        energy = feature.graycoprops(glcm, 'energy')[0, 0]
        correlation =feature. graycoprops(glcm, 'correlation')[0, 0]

        return area, perimeter, compactness, aspect_ratio, contrast, dissimilarity, homogeneity, energy, correlation,enclosing_circle_center_x,enclosing_circle_center_y,enclosing_circle_diameter
        # return  aspect_ratio
    def extract_cell_features(cell_image):
        cell_features = []
        contours , gray_cell_image ,a,binary_mask, filled_mask ,result_image = segmentation_cell_image(cell_image)
        for contour in contours:
            features = extract_features(contour, gray_cell_image)

            gray_hist = cv2.calcHist([gray_cell_image], [0], None, [256], [0, 256])
            gray_hist /= gray_hist.sum()

            cell_feature = np.concatenate((features, gray_hist.flatten()))
            cell_features.append(features)

        return cell_features , contours , a ,binary_mask, filled_mask ,result_image
    def test_image_for_cancer(image_path, svm_model):
    # image = preprocess_image(image_path)
        cell_features,contours, a, binary_mask, filled_mask , result_image = extract_cell_features(image_path)
        cell_predictions = svm_model.predict(cell_features)
        cancerous_cells = [contours[i] for i, prediction in enumerate(cell_predictions) if prediction == 1]
        cancer_percentage = np.mean(cell_predictions) * 100
        return cancer_percentage ,cancerous_cells,a,binary_mask, filled_mask ,result_image
    # Appliquer la segmentation (exemple avec seuillage
    # Charger le modèle SVM depuis le fichier enregistré
    save_path = "apps/home/SVM_ACC_99.pkl"
    loaded_model = joblib.load(save_path)

    # Fonction pour tester si une image contient des cellules cancéreuses
    cancer_percentage, cancerous_cells, a, binary_mask, filled_mask, result_image = test_image_for_cancer(cell_image, loaded_model)
    if cancer_percentage > 1:
        # Dessiner les cellules cancéreuses sur l'image
        for contour in cancerous_cells:
            # Utilisez les contours de chaque cellule pour dessiner un rectangle autour d'elle
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(cell_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    _, img_encoded = cv2.imencode('.jpg', cell_image)
    contours_img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return jsonify({'image': contours_img_base64})
