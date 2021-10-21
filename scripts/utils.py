import os
from datetime import datetime
import numpy as np
import pathlib
import tensorflow as tf
import tensorflow.keras as K
import time

def show_images(ds):
    plt.figure(figsize=(10, 10))
    #x,y=next(ds)
    for images in ds.next():
        for i in range(9):
            vis_img = images[i]
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(vis_img)
            plt.axis("off")
        plt.show()
        break

def get_all_files(root_path):
    files=[]
    for path, subdirs, files in os.walk(root_path):
        for name in files:
            files.append(os.path.join(path, name))
    return files




def predict_one(model,input_path,image_size=(100,100)):
    img = K.preprocessing.image.load_img(input_path, target_size=image_size)
    img_array = K.preprocessing.image.img_to_array(img)/255.0
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions,axis=1)
    return predicted_class[0],predictions[0,predicted_class]



def lite_average_time(interpreter,steps=500):
  times=[]
  for i in range(500):
    t1=time.time()*1000
    interpreter.invoke()
    t2=time.time()*1000
    times.append(t2-t1)
  print(f"Average time taken is {np.mean(times)}")


def pred_one_lite(interpreter_path,img_pth):
  interpreter = tf.lite.Interpreter(model_path=str(interpreter_path))
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]
  output_shape = interpreter.get_output_details()[0]["shape"]
  interpreter.allocate_tensors()
  tst_img=cv2.imread(img_pth)
  tst_img=tst_img[:,:,[2,1,0]]
  tst_img=np.expand_dims(tst_img,0)
  tst_img=tst_img.astype(np.float32)
  tst_img/=255.0
  interpreter.set_tensor(input_index, tst_img)
  interpreter.invoke()
  output = interpreter.tensor(output_index)
  digit = np.argmax(output(),axis=1)
  print(digit)


def time_lite(interpreter_path,img_pth):
  interpreter = tf.lite.Interpreter(model_path=str(interpreter_path))
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]
  output_shape = interpreter.get_output_details()[0]["shape"]
  interpreter.allocate_tensors()
  tst_img=cv2.imread(img_pth)
  tst_img=tst_img[:,:,[2,1,0]]
  tst_img=np.expand_dims(tst_img,0)
  tst_img=tst_img.astype(np.float32)
  tst_img/=255.0
  interpreter.set_tensor(input_index, tst_img)
  t1=time.time()*1000
  interpreter.invoke()
  t2=time.time()*1000
  print(f"The time taken is {t2-t1}")
  output = interpreter.tensor(output_index)
  digit = np.argmax(output(),axis=1)
  print(digit)



def evaluate_lite_model(interpreter_path,test_data):
  interpreter = tf.lite.Interpreter(model_path=str(interpreter_path))
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]
  output_shape = interpreter.get_output_details()[0]["shape"]
  # Resize input tensor to take 150 images batch size
  input_shape=[150,100,100,3]
  interpreter.resize_tensor_input(input_index,input_shape)
  interpreter.resize_tensor_input(output_index,[150, 1, output_shape[1]])  
  interpreter.allocate_tensors()
  # Run predictions on every image in the "test" dataset.
  prediction = []
  gt=[]
  print(f"Total test images batches {len(test_data)}")
  for i,(test_image,labels) in enumerate(test_data):
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    if i==len(test_data)-1:
        break
    test_image = test_image.astype(np.float32)
    interpreter.set_tensor(input_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output(),axis=1)
    prediction.extend(digit)
    gt.extend(np.argmax(labels,1))
    #print(f"Procesed {i} batches")
    #if i==20:
    #    break


  # Compare prediction results with ground truth labels to calculate accuracy.
  assert len(gt)==len(prediction), print("Length of predictions and GT are not equal")
  accurate_count = 0
  for index in range(len(prediction)):
    if prediction[index] == gt[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(prediction)
  
  return accuracy
