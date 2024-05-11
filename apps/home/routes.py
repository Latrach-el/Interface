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
import numpy as np
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
    filtered_img = cv2.bilateralFilter(img, 9, 75, 75)
    # Convertir l'image filtrée en base64 pour l'envoyer au frontend
    _, img_encoded = cv2.imencode('.jpg', filtered_img)
    filtered_img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return jsonify({'image': filtered_img_base64})
@blueprint.route('/lab_image', methods=['POST'])
def lab_image():
    # Récupérer le fichier d'image à partir de la requête
    file = request.files['file']
    # Lire l'image
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    # Appliquer le filtre (exemple avec flou gaussien)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Convertir l'image filtrée en base64 pour l'envoyer au frontend
    _, img_encoded = cv2.imencode('.jpg', lab_img)
    filtered_img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return jsonify({'image': filtered_img_base64})
# Route pour la segmentation de l'image
@blueprint.route('/segment_image', methods=['POST'])
def segment_image():
    # Récupérer le fichier d'image à partir de la requête
    file = request.files['file']
    # Lire l'image
    cell_image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.COLOR_BGR2RGB)
    cell_image = cv2.bilateralFilter(cell_image, 9, 75, 75)
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
    _, img_encoded = cv2.imencode('.png', filled_mask)
    segmented_img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return jsonify({'image': segmented_img_base64})
@blueprint.route('/masque_image', methods=['POST'])
def masque_image():
    # Récupérer le fichier d'image à partir de la requête
    file = request.files['file']
    # Lire l'image
    cell_image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.COLOR_BGR2RGB)
    # Appliquer la segmentation (exemple avec seuillage)
    filtered_img = cv2.bilateralFilter(cell_image, 9, 75, 75)
    image_lab = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2LAB)
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
@blueprint.route('/contour_image', methods=['POST'])
def contour_imagre():
    # Récupérer le fichier d'image à partir de la requête
    file = request.files['file']
    # Lire l'image
    cell_image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    filtered_img = cv2.bilateralFilter(cell_image, 9, 75, 75)
    # Appliquer la segmentation (exemple avec seuillage)
    image_lab = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2LAB)
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
@blueprint.route('/cancer_cell', methods=['POST'])
def contour_image():
    # Récupérer le fichier d'image à partir de la requête
    file = request.files['file']
    # Lire l'image
    cell_image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    cell_image = cv2.bilateralFilter(cell_image, 9, 75, 75)
    def preprocess_image(image_path):
    # Charger l'image
        # blood_image = cv2.imread(image_path)
    # Convertir l'image en Lab
        image_lab = cv2.cvtColor(image_path, cv2.COLOR_BGR2LAB)

    # Extraire la composante a* de l'image LAB
        _, component_a, _ = cv2.split(image_lab)

        return component_a,image_path,image_lab
# Fonction qui segmenter l'image
    # Fonction qui segmenter l'image
    def segmentation_blood_image(blood_image,component_a):

# Appliquer un seuillage d'Otsu
        _, binary_mask = cv2.threshold(component_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Remplir les trous dans le masque
        filled_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        filled_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

# Appliquer le masque sur l'image d'origine
        result_image = cv2.bitwise_and(blood_image, blood_image, mask=filled_mask)

# Convertir l'image en niveaux de gris
        gray_blood_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

# Trouver les contours dans l'image en niveaux de gris
        contours, _ = cv2.findContours(gray_blood_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours , gray_blood_image,component_a,binary_mask, filled_mask ,result_image
# Définir une fonction pour extraire les caractéristiques morphologiques et texturales
    # Fonction pour extraire les caractéristiques morphologiques, de texture, de couleur et de voisinage
    def extract_features(contour, gray_image,color_image):

        # Extraire les caractéristiques morphologiques
        area = cv2.contourArea(contour)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        enclosing_circle_center_x, enclosing_circle_center_y = (int(x), int(y))
        enclosing_circle_diameter = 2 * int(radius)
        perimeter = cv2.arcLength(contour, True)
        compactness = 4 * np.pi * area / perimeter**2 if perimeter > 0 else 0
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        x, y, w, h = cv2.boundingRect(contour)
        # glcm = greycomatrix(image, [2], [0], 256, symmetric=True, normed=True)
        cell_roi_color = color_image[y:y+h, x:x+w]
        lab_image = cv2.cvtColor(cell_roi_color, cv2.COLOR_BGR2LAB)

    # Calculer les moyennes des composantes LAB
        lab_mean = cv2.mean(lab_image)

    # Extraire les moyennes des composantes LAB
        L_mean = lab_mean[0]
        A_mean = lab_mean[1]
        B_mean = lab_mean[2]

    # Extraire les caractéristiques de couleur pour la cellule
        color_mean = cv2.mean(cell_roi_color)
        cell_roi = gray_image[y:y+h, x:x+w]
        glcm = feature.graycomatrix(cell_roi, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
        energy = feature.graycoprops(glcm, 'energy')[0, 0]
        correlation =feature. graycoprops(glcm, 'correlation')[0, 0]
        entropy = -np.sum(glcm * np.log2(glcm + np.finfo(float).eps))

    # Calculate rectangularity
        major_axis = max(w, h)
        minor_axis = min(w, h)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        rectangularity = area / (major_axis * minor_axis)

    # Calculate solidity
        solidity = area / (perimeter * perimeter) if perimeter>0 else 0

   # Calculate circularity
        circularity = (4 * np.pi * area) / perimeter if perimeter>0 else 0

    # Calculate convexity
        hull = cv2.convexHull(contour)
        convexity_area = cv2.contourArea(hull)
        convexity = convexity_area / area if area>0 else 0

    # Calculate eccentricity
        eccentricity = 0.5 * (major_axis - minor_axis) / major_axis if major_axis>0 else 0

    # Calculate elongation (aspect ratio)
        elongation = float(max(w, h)) / min(w, h)

        return area, perimeter, compactness, elongation, contrast, dissimilarity, homogeneity, energy, correlation, enclosing_circle_center_x, enclosing_circle_center_y, enclosing_circle_diameter,entropy,B_mean,A_mean,L_mean,rectangularity,solidity,eccentricity,circularity,convexity

        # return  aspect_ratio
    def extract_cell_features(blood_image,component_a):
        cell_features = []
        contours , gray_blood_image ,component_a,binary_mask, filled_mask ,result_image = segmentation_blood_image(blood_image,component_a)
        for contour in contours:
            features = extract_features(contour, gray_blood_image,blood_image)

            cell_features.append(features)

        return cell_features , contours , component_a ,binary_mask, filled_mask ,result_image
    def test_image_for_cancer(image_path, svm_model):
        component_a,blood_image,image_lab = preprocess_image(image_path)
        cell_features,contours, a, binary_mask, filled_mask , result_image = extract_cell_features(blood_image,component_a)
        cell_predictions = svm_model.predict(cell_features)
        cancerous_cells = [contours[i] for i, prediction in enumerate(cell_predictions) if prediction == 1]
        cancer_percentage = np.mean(cell_predictions) * 100
        return cancer_percentage ,cancerous_cells,component_a,binary_mask, filled_mask ,result_image,blood_image,image_lab

    # Appliquer la segmentation (exemple avec seuillage
    # Charger le modèle SVM depuis le fichier enregistré
    save_path = "apps/home/SVM.pkl"
    loaded_model = joblib.load(save_path)

    # Fonction pour tester si une image contient des cellules cancéreuses
    cancer_percentage,cancerous_cells,component_a,binary_mask, filled_mask ,result_image,blood_image,image_lab = test_image_for_cancer(cell_image, loaded_model)
    if cancer_percentage > 1:
    # Dessiner les cellules cancéreuses sur l'image
        for contour in cancerous_cells:
            # Utilisez les contours de chaque cellule pour dessiner un rectangle autour d'elle
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(cell_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    _, img_encoded = cv2.imencode('.jpg', cell_image)
    contours_img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return jsonify({'image': contours_img_base64})
