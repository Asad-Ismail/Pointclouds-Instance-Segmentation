
from __future__ import print_function

import os
import json
from io import StringIO
import sys
import signal
import traceback
import h5py
import flask
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import tensorflow.keras as K
import argparse
import numpy as np
import time
from Model import Model

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

#clf = load_model(os.path.join(model_path, 'keras.h5'),compile=True)

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded
    @classmethod
    def get_model(cls):
         """Get the model object for this instance, loading it if it's not already loaded."""
         if cls.model == None:
            print(f"Initilaizing the model") 
            cls.model=Model()
            cls.model.load_weights(os.path.join(model_path,"weights_best.hdf5"))
            #cls.model.load_weights("/home/asad/projs/fruits_classification/fruits-class/weights_best.hdf5")
         return cls.model

    @classmethod
    def predict(cls, img):
        img_array = tf.expand_dims(img, 0)
        predictions = cls.model.predict(img_array)
        predicted_class = np.argmax(predictions,axis=1)
        return predicted_class[0],predictions[0,predicted_class]



# The flask app for serving predictions
#app = flask.Flask(__name__)

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
    img=np.array(pay_load["images"], dtype='float32')
    #Preproceessing the image could be handeled in the predcition model
    img=img/255.0
    # Do the prediction
    cls_pred,score=ScoringService.predict(img)
    response={"class":cls_pred.item(),"score":score.item()}
    result=json.dumps(response)
    print(f"Encodded json dump is {result}")
    return flask.Response(response=result, status=200, mimetype="application/json")
