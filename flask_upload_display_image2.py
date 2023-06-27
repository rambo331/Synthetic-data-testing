from flask import Flask, render_template, request, session
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

# Define the path to the Keras model
MODEL_PATH = 'model_mixed_data.keras'


# Define the upload folder path
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')

# Create Flask application
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')

# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'

# Load the Keras model
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index_upload_and_display_image.html')

@app.route('/', methods=['POST'])
def uploadFile():
    if 'file' in request.form:
        # Image captured from camera
        img_data = request.form['file']
        img = preprocess_image(img_data)
    else:
        # Uploaded file
        uploaded_img = request.files['uploaded-file']
        if uploaded_img.filename == '':
            return "No file selected"
        
        # Extracting uploaded data file name
        img_filename = secure_filename(uploaded_img.filename)
        
        # Upload file to the upload folder
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        
        # Preprocess the uploaded image
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        img = preprocess_image(img_path)

    # Make predictions using the loaded model
    predictions = model.predict(np.expand_dims(img, axis=0))
    # Process the predictions as needed
    i = list(predictions.reshape(6))
    i = i.index(max(i))
    if i==0:
        predictions = 'PlayStation\(PS\)'
    else:
        predictions = 'Xbox'
    print(predictions)    

    # Render the results on the webpage
    return render_template('results.html', predictions=predictions)

def preprocess_image(img_path):
    # Preprocess the image using OpenCV or any other image processing library
    img = cv2.imread(img_path)
    # Adjust the preprocessing steps according to your model's requirements
    IMAGE_SIZE = (150,150)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    return(img)

if __name__ == '__main__':
    app.run(debug=True)
