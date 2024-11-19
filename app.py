from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pyngrok import ngrok
import base64
import io

# Configuración de Matplotlib
matplotlib.use('Agg')

# Crear la aplicación Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')

# Crear el directorio para subir archivos si no existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Función para generar imagen con puntos clave
def generate_image_with_keypoints(image, faces):
    """
    Dibuja puntos clave sobre los rostros detectados en la imagen.
    """
    output_img = image.copy()
    for (x, y, w, h) in faces:
        # Dibujar un rectángulo alrededor del rostro
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Convertir la imagen a formato PNG en memoria
    _, buffer = cv2.imencode('.png', output_img)
    return io.BytesIO(buffer)

# Página principal con el formulario para subir imágenes
@app.route('/')
def index():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('index.html', images=images)

# Ruta para subir y analizar la imagen
@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        # Verificar si se subió un archivo
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No se ha subido ninguna imagen.'}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img = cv2.imread(filepath)
        elif 'existing_file' in request.form:
            filename = request.form['existing_file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = cv2.imread(filepath)
        else:
            return jsonify({'error': 'No se ha proporcionado ninguna imagen.'}), 400

        # Convertir la imagen a escala de grises y detectar rostros
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return jsonify({'error': 'No se detectaron rostros en la imagen.'}), 400

        # Generar la imagen con puntos clave
        output = generate_image_with_keypoints(gray_img, faces)

        # Convertir la imagen generada a base64
        encoded_image = base64.b64encode(output.getvalue()).decode('utf-8')
        return jsonify({'image': encoded_image})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ruta para aplicar transformaciones a la imagen
@app.route('/transform', methods=['POST'])
def transform_image():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No se ha subido ninguna imagen.'}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img = cv2.imread(filepath)
        elif 'existing_file' in request.form:
            filename = request.form['existing_file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = cv2.imread(filepath)
        else:
            return jsonify({'error': 'No se ha proporcionado ninguna imagen.'}), 400

        # Voltear horizontalmente, aumentar brillo y voltear verticalmente
        flipped_img = cv2.flip(img, 1)
        bright_img = cv2.convertScaleAbs(flipped_img, alpha=1.2, beta=50)
        transformed_img = cv2.flip(bright_img, 0)

        # Guardar la imagen transformada en memoria
        _, buffer = cv2.imencode('.png', transformed_img)
        output = io.BytesIO(buffer)

        # Convertir a base64
        encoded_image = base64.b64encode(output.getvalue()).decode('utf-8')
        return jsonify({'image': encoded_image})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ruta para servir los archivos subidos
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Ejecutar la aplicación Flask
if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f" * ngrok URL: {public_url}")
    app.run(port=5000)
