import cv2
import paho.mqtt.client as mqtt
from PIL import Image
import sys
import os
import urllib
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import numpy as np
import time
#from tf_trt_models.detection import download_detection_model, build_detection_graph



# initialize the broker
broker = "mqtt.eclipse.org"
#broker="172.18.0.2"
topic="hw3"

# Download SSD model for face detection
# https://github.com/yeephycho/tensorflow-face-detection
FROZEN_GRAPH_NAME = 'tensorflow-face-detection/model/frozen_inference_graph_face.pb'
#!wget https://github.com/yeephycho/tensorflow-face-detection/blob/master/model/frozen_inference_graph_face.pb?raw=true -O {FROZEN_GRAPH_NAME}

# Load the Frozen graph

output_dir=''
frozen_graph = tf.GraphDef()
with open(os.path.join(output_dir, FROZEN_GRAPH_NAME), 'rb') as f:
  frozen_graph.ParseFromString(f.read())

# Set up for Face Detection
INPUT_NAME='image_tensor'
BOXES_NAME='detection_boxes'
CLASSES_NAME='detection_classes'
SCORES_NAME='detection_scores'
MASKS_NAME='detection_masks'
NUM_DETECTIONS_NAME='num_detections'
DETECTION_THRESHOLD = 0.5

input_names = [INPUT_NAME]
output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50)
                                                                                                                                            
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

tf_sess = tf.Session(config=tf_config)

# use this if you want to try on the optimized TensorRT graph
# Note that this will take a while
# tf.import_graph_def(trt_graph, name='')

# use this if you want to try directly on the frozen TF graph
# this is much faster
tf.import_graph_def(frozen_graph, name='')

tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')


def on_publish(client, message, result):
     print("message was published")

# initialize client
client = mqtt.Client("admin")
client.on_publish = on_publish
client.connect(broker,1883,60000)
print ("client connected")

# start capturing bode from usb webcam (Device 0 in my case)
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

j=0

while(True):
        ret,frame=cap.read()

        # resize for model
        image=cv2.resize(frame,(300,300))
#        _, img = cv2.imencode('.png',image)
#       msg = img.tobytes()

#       client.publish(topic, msg , 0)
        #cv2.imshow("test",image)
        print ("after raw image")

        # run face detection network
        scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={tf_input: image[None, ...]})

        boxes = boxes[0] # index by 0 to remove batch dimension
        scores = scores[0]
        classes = classes[0]
        num_detections = num_detections[0]
        print ("num detection:", num_detections)

        # plot boxes exceeding score threshold
        for i in range(int(num_detections)):
            if scores[i] < DETECTION_THRESHOLD:
                continue
             # scale box to image coordinates
            box = boxes[i] * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])
            x=int(box[1])
            y=int(box[0])
            w=int(box[3]-box[1])
            h=int(box[2]-box[0])
            print ("face detected")

            print(x,y,w,h)

            face=image[y:y+h,x:x+w]
            print ("face", face)
            _,png =cv2.imencode('.png',face)
            msg=png.tobytes()
            print ("msg to bytes", msg)
            #cv2.imshow('frame',face)
            #cv2.imwrite("file1.png",png)

            #publish to the broker
            client.publish(topic,payload= msg,qos= 0)
            print ("message published")

     # if cv2.waitKey(1) & 0xFF == ord('q'):
      #    break
        if j > 10:
              break
        j += 1

cap.release()
cv2.destroyAllWindows()
