import os
import numpy as np
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Memastikan folder upload ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Memuat model (Lazy loading bisa dipertimbangkan, tapi untuk demo kita load di awal)
# Pastikan file model_padi.h5 ada di direktori yang sama
MODEL_PATH = 'model_padi.h5'
model = None

def load_prediction_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            print("Model berhasil dimuat.")
        except Exception as e:
            print(f"Gagal memuat model: {e}")
    else:
        print(f"File model '{MODEL_PATH}' tidak ditemukan. Pastikan file model ada.")

# Daftar kelas penyakit
CLASS_NAMES = ["blast", "brownspot", "healthy", "hispa", "leafsmut", "tungro"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    # Memuat gambar dengan ukuran target 224x224
    img = image.load_img(img_path, target_size=(224, 224))
    # Konversi ke array numpy
    img_array = image.img_to_array(img)
    # Menambah dimensi batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalisasi (0-1) seperti saat training (biasanya MobileNetV2 pakai preprocessing specific, tapi user minta normalisasi 0-1 atau standar)
    # User request: "normalisasi" -> asumsi rescale 1./255
    img_array = img_array / 255.0
    return img_array

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="Tidak ada file yang diunggah")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error="Tidak ada file yang dipilih")
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if model is None:
            load_prediction_model()
            if model is None:
                return render_template('index.html', error="Model tidak ditemukan. Harap unggah model_padi.h5 ke server.", image_url=url_for('static', filename=f'uploads/{filename}'))

        try:
            # Preprocessing
            processed_img = preprocess_image(filepath)
            
            # Prediksi
            predictions = model.predict(processed_img)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[predicted_class_index]
            confidence = float(predictions[0][predicted_class_index]) * 100
            
            return render_template('index.html', 
                                   prediction=predicted_class, 
                                   confidence=f"{confidence:.2f}", 
                                   image_url=url_for('static', filename=f'uploads/{filename}'))
        
        except Exception as e:
            return render_template('index.html', error=f"Terjadi kesalahan saat memproses gambar: {e}")
            
    return render_template('index.html', error="Format file tidak didukung. Gunakan PNG, JPG, atau JPEG.")

if __name__ == '__main__':
    load_prediction_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
