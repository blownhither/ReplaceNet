import base64
import skimage
from skimage.io import imread
from flask import Flask, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


def view_base64_image(base64_string):
    from matplotlib import pyplot as plt

    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")

    imgdata = base64.b64decode(base64_string)
    img = imread(imgdata, plugin='imageio')
    print(img)
    plt.imshow(img)
    plt.show()


@app.route('/', methods=['POST'])
def test_post():
    # print(request.files)
    # print(request.form)
    # print(type(request.form['background']))
    # print(request.json)
    view_base64_image(request.form['background'][len('data:image/png;base64,'):])
    view_base64_image(request.form['mask'][len('data:image/png;base64,'):])
    return 'OK'


if __name__ == '__main__':
    app.run(port=2951, debug=True)

