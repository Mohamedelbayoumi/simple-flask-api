from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np


loaded_model = tf.keras.models.load_model('mobilenet_natural_images.keras')
class_names = ['street', 'forest', 'glacier', 'mountain', 'sea', 'building']


app = Flask(__name__)


def process_image(img_bytes):

        img = tf.convert_to_tensor(img_bytes)
        img = tf.image.decode_jpeg(img, channels=3)
        predicted_class = classify_image(img)
        return predicted_class


def classify_image(image):
        resize = tf.image.resize(image, (224, 224))
        resize = tf.cast(resize, tf.float32) / 255.0
        resize = tf.expand_dims(resize, axis=0)

        yhat = loaded_model.predict(resize)
        max_index = np.argmax(yhat)
        predicted_class = class_names[max_index]  # Convert max_index to integer scalar
        
        return predicted_class



@app.route('/search_image', methods=['POST'])
def search_image():
    data = request.json
    img_buffer = data.get('image')
    
    if not img_buffer:
        return jsonify({"error": "No image buffer provided"}), 400

    try :
        img_bytes = bytes(img_buffer)

        result = process_image(img_bytes)

        return jsonify({"caterogy": result}), 200


    except Exception as e:
            return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
