const IMAGE_SIZE = 256;

function crop_and_resize(image, size) {
    // given a Konva image, crop it to 1:1 and then resize to (size, size)
    // the change works in-place
    let original_size = image.size();
    let height = original_size.height, width = original_size.width;
    if(height > width) {
        image.crop({
            x: 0,
            y: (height - width) / 2,
            width: width,
            height: width,
        });
    } else if(height < width) {
        image.crop({
            x: (width - height) / 2,
            y: 0,
            width: height,
            height: height,
        });
    }
    image.height(size);
    image.width(size);
}

$(document).ready(function () {
    // Adapted from Konva tutorial
    let container = $('#pad_container');
    let stage_size = Math.min(container.width(), IMAGE_SIZE);
    let width = stage_size;
    let height = stage_size;
    console.log('stage_size', stage_size);
    let strokes = [];
    let currentStroke = [];

    // first we need Konva core things: stage and layer
    let stage = new Konva.Stage({
        container: 'pad_container',
        width: width,
        height: height
    });
    let background_layer = new Konva.Layer();   // show image
    stage.add(background_layer);

    let layer = new Konva.Layer();
    stage.add(layer);
    layer.alpha(0.5);

    background_layer.setZIndex(0);
    layer.setZIndex(1);

    // then we are going to draw into special canvas element
    let canvas = document.createElement('canvas');
    // canvas.width = stage.width();         // TODO: originally / 2
    // canvas.height = stage.height();
    canvas.width = width;
    canvas.height = height;

    // created canvas we can add to layer as "Konva.Image" element
    let image = new Konva.Image({
        image: canvas,
        x: (stage.width() - canvas.width) / 2,
        y: (stage.height() - canvas.height) / 2,
    });
    layer.add(image);
    stage.draw();

    // Good. Now we need to get access to context element
    let context = canvas.getContext('2d');
    context.strokeStyle = "#b8b8b8";
    context.lineJoin = "round";
    context.lineWidth = 25;

    let isPaint = false;
    let lastPointerPosition;
    let mode = 'brush';

    // now we need to bind some events
    // we need to start drawing on mousedown
    // and stop drawing on mouseup
    image.on('mousedown touchstart', function () {
        isPaint = true;
        lastPointerPosition = stage.getPointerPosition();
    });

    // will it be better to listen move/end events on the window?
    stage.addEventListener('mouseup touchend', function () {
        if (isPaint && currentStroke.length > 0) {
            strokes.push(currentStroke);
            currentStroke = [];
            console.log(strokes);
        }
        isPaint = false;
    });

    // and core function - drawing
    stage.addEventListener('mousemove touchmove', function () {
        if (!isPaint) {
            return;
        }

        if (mode === 'brush') {
            context.globalCompositeOperation = 'source-over';
        }
        if (mode === 'eraser') {
            context.globalCompositeOperation = 'destination-out';
        }
        context.beginPath();

        let localPos = {
            x: lastPointerPosition.x - image.x(),
            y: lastPointerPosition.y - image.y()
        };
        context.moveTo(localPos.x, localPos.y);
        let pos = stage.getPointerPosition();
        localPos = {
            x: pos.x - image.x(),
            y: pos.y - image.y()
        };
        context.lineTo(localPos.x, localPos.y);

        currentStroke.push(localPos);

        context.closePath();
        context.stroke();


        lastPointerPosition = pos;
        layer.batchDraw();
    });

    function _reset_mask() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        layer.batchDraw();
    }

    (function _bind_button() {
        let backgroud_image_copy = null;
        $('#upload_button').change(function (event) {
            backgroud_image_copy = Konva.Image.fromURL(URL.createObjectURL(event.target.files[0]), function (image) {
                background_layer.removeChildren();
                crop_and_resize(image, IMAGE_SIZE);
                background_layer.add(image);
                background_layer.draw();
                _reset_mask();
            });
        });

        $('#submit_button').click(function () {
            background_layer.toImage({
                callback(background) {
                    layer.toImage({
                        callback(mask) {
                            // let form = new FormData();
                            // form.append('background', background);
                            // form.append('mask', mask);
                            $.ajax({
                                url: "http://localhost:2951",
                                method: "POST",
                                data: {
                                    background: background.src, // string, base64
                                    mask: mask.src,
                                },
                                dataType: 'json'
                                // processData: false,
                                // contentType: false,
                            });
                        }
                    });
                }
            });


            // let image = $('#upload_button')[0].files[0];

        });
    })();
});