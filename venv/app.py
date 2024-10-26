from flask import Flask, request, render_template, redirect, url_for
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Simulación de un DataFrame con imágenes (reemplaza esto con tus datos reales)
keyfacial_df = pd.DataFrame({
    'Image': [np.random.rand(64, 64) for _ in range(100)]  # Ejemplo de imágenes aleatorias
})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    # Guardar la imagen original
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Procesar la imagen cargada
    processed_image_filename = process_image(filepath)

    # Asegurarse de que el archivo se guardó correctamente
    if processed_image_filename:
        print(f'Processed image saved as: {processed_image_filename}')
    else:
        print("Error: No se pudo procesar la imagen.")

    # Renderizar el template con la imagen original y procesada
    return render_template('index.html', image=file.filename, processed_image=processed_image_filename)

def process_image(filepath):
    try:
        # Código de procesamiento de la imagen
        fig = plt.figure(figsize=(20, 20))

        for i in range(64):
            k = random.randint(0, len(keyfacial_df) - 1)  # Asegurarse de que k esté en el rango
            ax = fig.add_subplot(8, 8, i + 1)
            plt.imshow(keyfacial_df['Image'][k], cmap='gray')
            for j in range(1, 31, 2):
                plt.plot(keyfacial_df.loc[k][j-1], keyfacial_df.loc[k][j], 'rx')

        # Guardar la imagen procesada
        processed_image_filename = 'processed_' + os.path.basename(filepath)
        processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_image_filename)
        plt.savefig(processed_image_path)
        plt.close(fig)

        # Verificar que la imagen se ha guardado correctamente
        if os.path.exists(processed_image_path):
            return processed_image_filename  # Devuelve el nombre de la imagen procesada
        else:
            print("Error: La imagen procesada no se pudo guardar.")
            return None
            
    except Exception as e:
        print(f"Ocurrió un error durante el procesamiento de la imagen: {e}")
        return None

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
