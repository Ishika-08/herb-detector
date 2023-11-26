import pickle
from flask import Flask, request, render_template
import cv2
import numpy as np

app = Flask(__name__)

# Load the image classification model
with open('./models/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


def preprocess_image(image):
    resized_img = cv2.resize(image, (224, 224))
    # flattened_img = resized_img.flatten()  # Flatten the image to 1D
    flattened_img = np.array(resized_img).flatten()
    return flattened_img
   
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        # Get the image file from the POST request
        file = request.files['image']

        # Read the image file
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        # Preprocess the image for prediction
        processed_img = preprocess_image(img)

        # Make prediction using the loaded model
        prediction = model.predict(np.expand_dims(processed_img, axis=0))

        # Assuming 'categories' contains your class labels
        categories = ['Jasmine', 'Lemon', 'Mint', 'Neem', 'Peepal']
        result = categories[prediction[0]]

        return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
