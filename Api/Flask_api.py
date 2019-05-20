import base64
import io
import pickle
from io import StringIO, BytesIO

import tensorflow as tf
import keras
import keras.backend as K
import numpy as np
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import pyplot as plt
from PIL import Image
from flask import Flask, send_file, render_template, Response
import numpy as np
from flask_cors import CORS
from flask import jsonify
from flask import request
from keras.engine import Layer
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras_multi_head import MultiHeadAttention
from keras_preprocessing.sequence import pad_sequences
from keras.backend.tensorflow_backend import set_session
from matplotlib.ticker import NullLocator
from vis.visualization import visualize_cam
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

from metrics import auc_roc
from multiplicative_lstm import MultiplicativeLSTM


class Pentanh(Layer):
    def __init__(self, **kwargs):
        super(Pentanh, self).__init__(**kwargs)
        self.supports_masking = True
        self.__name__ = 'pentanh'

    def call(self, inputs):
        return K.switch(K.greater(inputs, 0), K.tanh(inputs), 0.25 * K.tanh(inputs))

    def get_config(self):
        return super(Pentanh, self).get_config()

    def compute_output_shape(self, input_shape):
        return input_shape

def get_model():
    global model
    print("Start loading")
    keras.utils.generic_utils.get_custom_objects().update({'pentanh': Pentanh()})
    model = load_model("weights-improvement-MERGED-11-0.11.hdf5",
                       custom_objects={'MultiplicativeLSTM': MultiplicativeLSTM,
                                       'MultiHeadAttention': MultiHeadAttention,
                                       'auc_roc': auc_roc}, compile=False)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', auc_roc])

    global graph
    graph = tf.get_default_graph()
    print(" * Model loaded!")

def preprocess_image(image, target_size):
    print("Start Preprocess")
    if image.mode != "L":
        image = image.convert("L")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image/255
    print("End preprocess")
    return image

def preprocess_image2(image, target_size):
    print("Start Preprocess")
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    print("End preprocess")
    return image

def preprocess_text(text):
    print("Load tokenizer")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    print("Preparing train data")
    text = tokenizer.texts_to_sequences(text)
    text = pad_sequences(text, maxlen=500, padding='post')

    return text

app = Flask(__name__)
CORS(app)

@app.route("/conv", methods=["POST"])
def comv():
    print("Start visualization")
    message = request.get_json(force=True)
    encoded = message['image']
    print("Start visualization")
    decoded = base64.b64decode(encoded)
    print("Start visualization")
    image = Image.open(io.BytesIO(decoded))
    print("Start visualization")
    plt.imshow(image)
    print(model.layers[2357].get_config())
    grads = visualize_cam(model, 2357, 13, image)
    plt.imshow(grads, cmap='jet', alpha=0.5)
    print("Send file")
    #img_io = BytesIO()
    #plt.savefig(img_io, format='png')
    #img_io.seek(0)
    #output = io.BytesIO()
    #FigureCanvas(plt.gcf()).print_png(output)
    #return Response(output.getvalue(), mimetype='image/png')
    figfile = BytesIO()
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())

    plt.savefig(figfile, format='png', bbox_inches = 'tight', pad_inches=0)
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())
    return send_file(io.BytesIO(figdata_png),
                     attachment_filename='logo.png',
                     mimetype='image/png')


@app.route("/predict", methods=["POST"])
def predict():
    print("Start Pred")
    message = request.get_json(force=True)
    encoded = message['image']
    text = message['text']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))
    processed_image = np.expand_dims(processed_image, axis=0)
    processed_text = preprocess_text(text)
    with graph.as_default():
        prediction = model.predict([processed_image,processed_text])
        print(prediction)

    response = {
        'prediction': {
            'No findings': str(prediction[0][0]),
            'Enlarged Cardiomediastinum': str(prediction[0][1]),
            'Cardiomegaly':str( prediction[0][2]),
            'Airspace Opacity': str(prediction[0][3]),
            'Lung Lesion': str(prediction[0][4]),
            'Edema': str(prediction[0][5]),
            'Consolidation': str(prediction[0][6]),
            'Pneumonia': str(prediction[0][7]),
            'Atelectasis': str(prediction[0][8]),
            'Pneumothorax': str(prediction[0][9]),
            'Pleural Effusion': str(prediction[0][10]),
            'Pleural Other': str(prediction[0][11]),
            'Fracture': str(prediction[0][12]),
            'Support Devices': str(prediction[0][13])
        }
    }
    print(response)
    print("End Pred")
    #print(model.layers[2357].get_config())
    #print(model.summary())
    #grads = visualize_cam(model, 2357, 13, image)

    return jsonify(response)


if __name__ == '__main__':
    app.debug=True
    print(" * Loading Keras model...")
    get_model()
    app.run()



'''
COMPARISON None  INDICATION: Hypertension FINDINGS Density in the left upper lung on PA XXXX XXXX represents superimposed bony and vascular structures. There is calcification of the first rib costicartilage junction which XXXX contributes to this appearance. The lungs otherwise appear clear. The heart and pulmonary XXXX appear normal. In the pleural spaces are clear. The mediastinal contour is normal. There are degenerative changes of thoracic spine. There is an electronic cardiac device overlying the left chest wall with intact distal leads in the right heart.  IMPRESSION 1. Irregular density in the left upper lung on PA XXXX, XXXX artifact related to superimposed vascular bony structures. Chest fluoroscopy or XXXX would confirm this 2.Otherwise, no acute cardiopulmonary disease.
'''