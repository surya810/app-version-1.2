from flask import Flask, render_template, request
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import keras.utils as image
import cv2
# from google.colab.patches import cv2_imshow
from PIL import Image
import pickle

app = Flask(__name__, template_folder='templates')

# Load the VGG16 model
with open('models/vgg16_model.pkl', 'rb') as n:
    vmodel = pickle.load(n) 
    
for layers in (vmodel.layers):
    layers.trainable = False

with open('models/hybrid_surya_model.pkl', 'rb') as f:
    pmodel = pickle.load(f)



def predict(image_path):
    image = Image.open(image_path)
    new_size = (224, 224)
    resized_image = image.resize(new_size)

    x = img_to_array(resized_image)
    x = x / 255
    x = np.expand_dims(x, axis=0)

    # Pass the input to the VGG16 model and get the output of the convolutional layers
    feature_extractor = vmodel.predict(x)

    # Flatten the output of the convolutional layers
    features = feature_extractor.reshape(feature_extractor.shape[0], -1)

    prediction = pmodel.predict(features)

    return prediction


@app.route('/')
def index():
    # image_file = 'static/img/bf.webp'
    # return render_template('index.html', image_file=image_file)
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from the request
        file = request.files['image']
        # Save the file to disk
        file_path = 'static/images/' + file.filename
        file.save(file_path)
        # Get the prediction for the file
        prediction = predict(file_path)
        # Prepare the response
        if prediction == 0:
            result = 'Benign'
        else:
            result = 'Malignant'
        return render_template('result.html', image=file_path, result=result)


if __name__ == '__main__':
    app.run(debug=True)
    