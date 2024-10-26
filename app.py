from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
import dlib
import os
from werkzeug.utils import secure_filename

# Inicializar la app Flask
app = Flask(__name__)

# Rutas de los archivos
PREDICTOR_PATH = 'csv/shape_predictor_68_face_landmarks.dat'
IMAGES_FOLDER = 'imagenes'
PROCESSED_FOLDER = 'imagenes_procesadas'

# Crear carpetas si no existen
os.makedirs(IMAGES_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Cargar el predictor y el detector de rostros
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Mapeo de landmarks a columnas
landmark_mapping = {
    'left_eye_center_x': 36, 'left_eye_center_y': 37,
    'right_eye_center_x': 42, 'right_eye_center_y': 43,
    'left_eye_inner_corner_x': 39, 'left_eye_inner_corner_y': 40,
    'left_eye_outer_corner_x': 36, 'left_eye_outer_corner_y': 37,
    'right_eye_inner_corner_x': 42, 'right_eye_inner_corner_y': 43,
    'right_eye_outer_corner_x': 45, 'right_eye_outer_corner_y': 46,
    'left_eyebrow_inner_end_x': 17, 'left_eyebrow_inner_end_y': 18,
    'left_eyebrow_outer_end_x': 19, 'left_eyebrow_outer_end_y': 20,
    'right_eyebrow_inner_end_x': 22, 'right_eyebrow_inner_end_y': 23,
    'right_eyebrow_outer_end_x': 24, 'right_eyebrow_outer_end_y': 25,
    'nose_tip_x': 30, 'nose_tip_y': 31,
    'mouth_left_corner_x': 48, 'mouth_left_corner_y': 49,
    'mouth_right_corner_x': 54, 'mouth_right_corner_y': 55,
    'mouth_center_top_lip_x': 51, 'mouth_center_top_lip_y': 52,
    'mouth_center_bottom_lip_x': 57, 'mouth_center_bottom_lip_y': 58
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No se subió ninguna imagen", 400

    file = request.files['image']
    if file.filename == '':
        return "No se seleccionó ninguna imagen", 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(IMAGES_FOLDER, filename)
    processed_image_path = os.path.join(PROCESSED_FOLDER, filename)

    # Guardar la imagen subida
    file.save(image_path)

    # Leer la imagen y convertir a escala de grises
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Detectar las caras en la imagen
    faces = face_detector(gray)

    # Procesar cada rostro detectado
    for face in faces:
        landmarks = shape_predictor(gray, face)

        # Dibujar solo los puntos especificados en landmark_mapping
        for column_name, landmark_index in landmark_mapping.items():
            x = landmarks.part(landmark_index).x
            y = landmarks.part(landmark_index).y
            cv2.drawMarker(color_image, (x, y), color=(0, 0, 255),
                           markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)

    # Guardar la imagen procesada
    cv2.imwrite(processed_image_path, color_image)

    # Redirigir al index y pasar el nombre del archivo procesado
    return redirect(url_for('show_image', filename=filename))

@app.route('/imagenes_procesadas/<filename>')
def show_image(filename):
    return render_template('show_image.html', filename=filename)

@app.route('/processed/<filename>')
def send_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
