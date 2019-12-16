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
from inference import InferenceHelper


app = Flask(__name__)
CORS(app)
# app.logger.set
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.addHandler(logging.FileHandler('app.log', mode='a'))


def decode_image(base64_string):
    imgdata = base64.b64decode(base64_string)
    img = skimage.img_as_float(imread(imgdata, plugin='imageio'))
    return img


def view_base64_image(base64_string):
    from matplotlib import pyplot as plt

    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")

    imgdata = base64.b64decode(base64_string)
    img = imread(imgdata, plugin='imageio')
    plt.imshow(img)
    plt.show()


@app.route('/', methods=['GET'])
def get_index():
    return send_file('index.html')


@app.route('/favicon.ico', methods=['GET'])
def get_favicon():
    return send_file('favicon.ico')


@app.route('/index.js', methods=['GET'])
def get_index_js():
    return send_file('index.js')


@app.route('/style.css', methods=['GET'])
def get_style_css():
    return send_file('style.css')


@app.route('/assets/<path:filename>')
def send_assets_file(filename):
    return send_from_directory('assets', filename, as_attachment=False)


def plot_six_graph(fg_image, fg_mask, bg_image, bg_mask, synthesized, result):
    from matplotlib import pyplot as plt
    plt.subplot(2, 6, 1)
    plt.imshow(fg_image)
    plt.subplot(2, 6, 2)
    plt.imshow(fg_mask * 255)
    plt.subplot(2, 6, 7)
    plt.imshow(bg_image)
    plt.subplot(2, 6, 8)
    plt.imshow(bg_mask * 255)
    plt.subplot(1, 3, 2)
    plt.imshow(synthesized)
    plt.subplot(1, 3, 3)
    plt.imshow(result)
    plt.show()


@app.route('/replace', methods=['POST'])
def test_post():
    # view_base64_image(request.form['background'][len('data:image/png;base64,'):])
    # view_base64_image(request.form['mask'][len('data:image/png;base64,'):])
    foreground_id = request.form['foreground_id']
    log_str = f'foreground_id={foreground_id}'
    app.logger.info(log_str)

    background_image = decode_image(request.form['background'][len('data:image/png;base64,'):])
    background_image = background_image[:, :, :3]   # drop alpha
    background_mask = decode_image(request.form['mask'][len('data:image/png;base64,'):])
    background_mask = background_mask[:, :, 0] > 0
    foreground_image = imread(f'assets/foreground-{foreground_id}-raw.png')[:, :, :3]   # drop alpha
    foreground_mask = np.load(f'assets/foreground-{foreground_id}-mask.npy', allow_pickle=True)

    synthesized, result = inf.replace(foreground_image, foreground_mask, background_image, background_mask,
                                      use_dilation=5)

    plot_six_graph(foreground_image, foreground_mask, background_image, background_mask,
                   synthesized, result)
    result = skimage.img_as_ubyte(result)

    # mat = skimage.img_as_ubyte(imread('assets/random.jpg'))
    start = time.time()
    r, g, b = np.split(result, 3, axis=2)
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
    model_path = 'tmp/model-20191203185450/model'
    inf = InferenceHelper(model_path, patch_size=256)
    app.run(host='0.0.0.0', port=80, debug=False)

