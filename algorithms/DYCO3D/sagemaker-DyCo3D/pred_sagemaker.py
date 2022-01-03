
import sagemaker
from sagemaker.predictor import json_serializer, json_deserializer, Predictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def visualize_masks(masks,img):
    """Apply the masks to the image."""
    vis_img=img.copy()
    from random import randint
    for i in range(masks.shape[0]):
        color = [randint(0, 255) for p in range(3)]
        for j in range(3):
            vis_img[:, :,j] = np.where(masks[i] == True,color[j],vis_img[:,:,j])
    return vis_img

parser = argparse.ArgumentParser()
parser.add_argument("endpoint",help="Provide the endname of the argument")

parser.add_argument("--image",help="Input imgage",default="local_test/test_dir/input/data/validation/21XZ2424_210715_130508_220270_001.jpg")
args = parser.parse_args()

endpoint_name=args.endpoint
predictor = Predictor(endpoint_name=endpoint_name, sagemaker_session=sagemaker.Session(),serializer=sagemaker.serializers.JSONSerializer(),
    deserializer=sagemaker.deserializers.JSONDeserializer())

#predictor = Predictor(endpoint_name=endpoint_name, sagemaker_session=sagemaker.Session())
image_path=args.image
img=cv2.imread(image_path)
assert img is not None, "Input Image is None Aborting!!"
img=cv2.resize(img,(int(img.shape[1]//2),int(img.shape[0])//2))
img.shape
plt.imshow(img[...,::-1])
plt.title("Input Image")

data={"images":img.tolist(),"batch_size":1}

res=predictor.predict(data)

res.keys()

detected_masks=visualize_masks(np.array(res["masks"]),img)

plt.imshow(detected_masks[...,::-1])