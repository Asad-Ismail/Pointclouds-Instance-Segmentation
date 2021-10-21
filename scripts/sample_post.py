import requests
import PIL
import json
from PIL import Image
import numpy as np

def post_image(img_file):
    """ post image and return the response """
    img = np.array(Image.open(img_file))
    data={"images":img.tolist(),"batch_size":1}
    json_data=json.dumps(data)
    img=np.array(json.loads(json_data)["images"])
    headers={}
    URL="http://127.0.0.1:8080/invocations"
    response = requests.post(URL, data=json_data, headers=headers)
    return response


post_image("/home/asad/projs/fruits_classification/fruits-class/test.jpg")