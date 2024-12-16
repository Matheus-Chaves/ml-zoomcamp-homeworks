import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
from tflite_runtime.interpreter import Interpreter

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def lambda_handler(event, context):
    url = event['url']
    img = download_image(url)
    img = prepare_image(img, (200, 200))
    image_array = np.array(img, dtype=np.float32) / 255.0

    interpreter = Interpreter(model_path='model_2024_hairstyle_v2.tflite')
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], np.expand_dims(image_array, axis=0))
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0][0]
    return float(output)