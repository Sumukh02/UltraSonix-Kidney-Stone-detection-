
from flask import Flask, render_template, redirect, url_for, request
import cv2
import numpy as np
from skimage.feature import hog
from keras.models import load_model 
import joblib

app = Flask(__name__)
from flask_bootstrap import Bootstrap
bootstrap = Bootstrap(app)

# Load the SVM model
svm_model = joblib.load("svc.pkl")

# Load the CNN model
cnn_model = load_model("kidney_stone_detection_model.h5")

# Define function to perform SVM prediction

def predict_svm(image):
    # Preprocess image and extract features (HOG)
    resized_image = cv2.resize(image, (64, 128))  # Resize image to standard size for HOG
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

    # Ensure that the feature vector has the correct dimensions
    features = features[:2400]  # Adjusted to match the expected feature vector length

    # Perform prediction using SVM model
    prediction = svm_model.predict([features])

    # Map numerical prediction to class label
    class_label = "Positive" if prediction[0] == 1 else "Negative"

    # Return the prediction result
    return "SVM Prediction: {}".format(class_label)


# Define function to perform CNN prediction
def predict_cnn(image):
    # Preprocess image (resize and normalize)
    resized_image = cv2.resize(image, (150, 150))  # Assuming input size required by CNN model
    normalized_image = resized_image / 255.0

    # Perform prediction using CNN model
    prediction = cnn_model.predict(np.array([normalized_image]))

    # Process prediction result (example: converting probability to class label)
    class_label = "Positive" if prediction[0][0] > 0.5 else "Negative"

    # Return the prediction result
    return "CNN Prediction: {}".format(class_label)

# Route to predict choice page
@app.route('/predict_choice', methods=['GET', 'POST'])
def predict_choice():
    if request.method == 'POST':
        choice = request.form['choice']
        if choice == 'svm':
            return redirect(url_for('predict_svm_page'))
        elif choice == 'cnn':
            return redirect(url_for('predict_cnn_page'))
    return render_template('predict_choice.html')

# Route to SVM prediction page
@app.route('/predict_svm_page', methods=['GET', 'POST'])
def predict_svm_page():
    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_file = request.files['file']

        # Read the image file
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Perform prediction using SVM model
        prediction = predict_svm(image)

        # Return the prediction result
        return render_template('prediction_result.html', prediction=prediction)
    return render_template('predict_svm_page.html')

# Route to CNN prediction page
@app.route('/predict_cnn_page', methods=['GET', 'POST'])
def predict_cnn_page():
    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_file = request.files['file']

        # Read the image file
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Perform prediction using CNN model
        prediction = predict_cnn(image)

        # Return the prediction result
        return render_template('prediction_result.html', prediction=prediction)
    return render_template('predict_cnn_page.html')

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/welcome')
def welcome():
    return render_template('prediction.html')

@app.route('/login',methods=['POST'])
def login():
    return redirect(url_for('welcome'))

if __name__ == '__main__':
    app.run(debug=True)

