import requests
import PIL
import json
from PIL import Image
import numpy as np
import cv2
import sys

def post_image(img_file):
    """ post image and return the response """
    img = cv2.imread(img_file)
    img = cv2.resize(img, (img.shape[1]//3,img.shape[0]//3), interpolation = cv2.INTER_AREA)
    #img = np.array(Image.open(img_file))
    data={"images":img.tolist(),"batch_size":1}
    json_data=json.dumps(data)
    print ("Estimated Input size: " + str(sys.getsizeof(json_data) / 1e6) + "MB")
    img=np.array(json.loads(json_data)["images"])
    headers={}
    URL="http://127.0.0.1:8080/invocations"
    response = requests.post(URL, data=json_data, headers=headers)
    return response


post_image("/media/asad/ADAS_CV/datasets_Vegs/pepper/images/21XZ2418_210715_124212_719242_001.jpg")