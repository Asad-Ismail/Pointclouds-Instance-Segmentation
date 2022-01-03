from __future__ import print_function
import os
import json
from io import StringIO
import sys
import signal
import traceback
import flask
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import argparse
import numpy as np
import time
from fruit_pred_utils import *

#Model path
prefix = '/opt/ml/model/'
model_path = os.path.join(prefix, 'model_final.pth')
print(f"The Model path exists {os.path.exists(model_path)}")

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded
    @classmethod
    def get_model(cls):
         """Get the model object for this instance, loading it if it's not already loaded."""
         if cls.model == None:
            print(f"Initilaizing the model for Inference!!") 
            cls.model=detectroninference(model_path,name_classes=["Pepp"])
         return cls.model

    @classmethod
    def predict(cls, img):
        img_array = tf.expand_dims(img, 0)
        predictions = cls.model.predict(img_array)
        predicted_class = np.argmax(predictions,axis=1)
        return predicted_class[0],predictions[0,predicted_class]



# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    model=ScoringService.get_model()
    health = model is not None  # You can insert a health check here
    status = 200 if health else 404
    return flask.Response(response=f'Sucessfully Pinged Health status {status}\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. """
    data = None
    model=ScoringService.get_model()
    if model is None:
        print("Model could not load")
        exit(-1)
    pay_load=json.loads(flask.request.data)
    #print(f"Json loaded file is {pay_load}")
    img=np.array(pay_load["images"], dtype=np.uint8)
    #print(f"Loaded Image is {img}")
    print(f"Running Inference!!")
    masks,boxes,classes=model.pred(img)
    response={"masks":masks.tolist(),"boxes":boxes.tolist(),"classes":classes.tolist()}
    result=json.dumps(response)
    #print(f"Encodded json dump is {result}")
    return flask.Response(response=result, status=200, mimetype="application/json")
