import sys
import cv2
import time
import base64
import skimage
import logging
import numpy as np
from skimage.io import imread
from flask import Flask, request, send_file, send_from_directory, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
# app.logger.set
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.addHandler(logging.FileHandler('app.log', mode='a'))


def view_base64_image(base64_string):
    from matplotlib import pyplot as plt

    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")

    imgdata = base64.b64decode(base64_string)
    img = imread(imgdata, plugin='imageio')
    # print(img)
    plt.imshow(img)
    plt.show()


@app.route('/', methods=['GET'])
def get_index():
    return send_file('index.html')


@app.route('/index.js', methods=['GET'])
def get_index_js():
    return send_file('index.js')


@app.route('/style.css', methods=['GET'])
def get_style_css():
    return send_file('style.css')


@app.route('/assets/<path:filename>')
def send_assets_file(filename):
    return send_from_directory('assets', filename, as_attachment=False)


@app.route('/replace', methods=['POST'])
def test_post():
    view_base64_image(request.form['background'][len('data:image/png;base64,'):])
    view_base64_image(request.form['mask'][len('data:image/png;base64,'):])
    foreground_id = request.form['foreground_id']
    print(foreground_id)

    mat = skimage.img_as_ubyte(imread('assets/random.jpg'))

    start = time.time()
    r, g, b = np.split(mat, 3, axis=2)
    mat = np.dstack([b, g, r])  # cv2 uses BGR order
    success, content = cv2.imencode('.png', mat)
    content = content.tobytes()
    log_str = f'imwrite takes {time.time() - start}'
    app.logger.warning(log_str)

    return jsonify({
        'img': base64.b64encode(content).decode(),
        'status': 'OK'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2951, debug=True)

