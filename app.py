from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Load the model
model = load_model('model/brain_tumor_model.h5')
expected_input_size = (128, 128)  # Change this based on your model input
print(f"Model input shape: {model.input_shape}")  # Debugging

# Class names
class_names = {
    0: 'notumor',
    1: 'meningioma',
    2: 'pituitary',
    3: 'glioma'
}
def allowed_file(filename):
    """Check if the uploaded file is an allowed image type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img_path):
    """Preprocess image to match model's expected input shape."""
    try:
        # Load image and resize to match model input
        img = image.load_img(img_path, target_size=expected_input_size)
        
        # Convert to array and expand dimensions
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize pixel values to [0,1]
        img_array = img_array / 255.0
        
        print(f"Processed image shape: {img_array.shape}")  # Debugging
        return img_array

    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded.")
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return render_template('index.html', error="No file selected.")
        
        # Check if file type is allowed
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the uploaded file
            file.save(filepath)
            
            # Preprocess and predict
            processed_image = preprocess_image(filepath)
            
            if processed_image is None:
                return render_template('index.html', error="Error processing image.")

            try:
                # Make a prediction
                predictions = model.predict(processed_image, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = round(100 * np.max(predictions[0]), 2)
                result = class_names.get(predicted_class, "Unknown")
                
                return render_template('index.html', 
                                       filename=filename, 
                                       prediction=result,
                                       confidence=confidence)
            
            except Exception as e:
                # Handle prediction errors
                error_msg = f"Error during prediction: {str(e)}"
                print(error_msg)
                return render_template('index.html', error=error_msg)
    
    return render_template('index.html')

@app.route('/about-model')
def about_model():
    return render_template('about_model.html')

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, port=8000)
