from flask import Flask, request, render_template
import os
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('brain_tumor_model.h5')
UPLOAD_FOLDER = 'static/uploads/'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def predict_tumor(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = img.reshape(1, 150, 150, 1) / 255.0
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    categories = ["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"]
    return categories[class_index], prediction[0][class_index]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', result="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', result="No selected file")
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            result, confidence = predict_tumor(file_path)
            return render_template('index.html', result=result, confidence=confidence, image=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)