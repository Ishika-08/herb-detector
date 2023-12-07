import pickle
from flask import Flask, request, render_template
import cv2
import numpy as np

app = Flask(__name__)

plant_info = {
    'Jasmine': {
        'description': 'Jasmine is a genus of shrubs and vines in the olive family (Oleaceae). It is known for its beautiful and highly fragrant flowers. Jasmine flowers are often white or yellow and are prized for their sweet, pleasing scent. These flowers are used in perfumes, teas, and various ceremonial practices in different cultures.',
        'image_path': '/static/jasmine.jpg' 
    },
    'Lemon': {
        'description': 'Lemon is a yellow citrus fruit known for its acidic and sour taste. It belongs to the Rutaceae family and is a rich source of vitamin C. Lemons are commonly used in culinary applications to add flavor to dishes, beverages, and desserts. They are also used for their juice, zest, and as a garnish in many recipes.',
        'image_path': '/static/lemon.jpg'  
    },
    'Mint': {
        'description': 'Mint is a perennial herb known for its refreshing aroma and flavor. Belonging to the Lamiaceae family, mint leaves are widely used in culinary dishes, beverages (like mint tea or mojitos), and for medicinal purposes. It has a cooling sensation and is popular for its ability to soothe digestion and freshen breath.',
        'image_path': '/static/mint.jpg'  
    },
    'Neem': {
        'description': 'Neem is a tree native to the Indian subcontinent and belongs to the mahogany family, Meliaceae. It is known for its numerous medicinal properties and has been used in traditional medicine for centuries. Neem extracts are used in skincare, dental products, agriculture as a natural pesticide, and in Ayurvedic medicine.',
        'image_path': '/static/neem.jpg'  
    },
    'Peepal': {
        'description': 'Peepal, also known as the sacred fig, is a large deciduous tree native to the Indian subcontinent and some parts of Southeast Asia. It holds great significance in Hinduism, Buddhism, and Jainism. Peepal trees are considered sacred and are often found near temples. Its heart-shaped leaves and the tree as a whole hold cultural and religious importance.',
        'image_path': '/static/peepal.jpg'  
    }
}


with open('./models/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


def preprocess_image(image):
    resized_img = cv2.resize(image, (224, 224))
    flattened_img = np.array(resized_img).flatten()
    return flattened_img
   
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        file = request.files['image']
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        processed_img = preprocess_image(img)
        prediction = model.predict(np.expand_dims(processed_img, axis=0))
        categories = ['Jasmine', 'Lemon', 'Mint', 'Neem', 'Peepal']
        predicted_class = categories[prediction[0]]

        plant_description = plant_info[predicted_class]['description']
        plant_image_path = plant_info[predicted_class]['image_path']

        return render_template('result.html', result=predicted_class, description=plant_description, image_path=plant_image_path)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
