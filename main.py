"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished +to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    inference_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    model = args.model
    DEVICE = args.device
    CPU_EXTENSION = args.cpu_extension

    ### TODO: Load the model through `inference_network` ###
    inference_network.load_model(model, CPU_EXTENSION, DEVICE )
    network_shape = inference_network.get_input_shape()
    
    ### TODO: Handle the input stream ###
    ### Checking for WebCam feed
    if args.input == 'CAM':
        input_received = 0
    ##Check for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') or args.input.endswith('.png'):
        one_image_state = True
        input_received = args.input
    ##Checking for video file
    else:
        input_received = args.input
        assert os.path.isfile(args.input), "file doesn't exist"
    ##Handling the input stream from videos
    cap =  cv2.VideoCapture(input_received)
    cap.open(input_received)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_shape = network_shape['image_tensor']

    #Declaring required variables
    request_id = 0 #For Async Inference
    #As there can be ni person detected in the initial frames
    #Assign zero to all the variables
    current_duration = 0
    duration_prev = 0
    counter_total = 0
    count_publish = 0
    last_count = 0
    counter = 0 #For detecting the start when person enters frame
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame, (input_shape[3], input_shape[2]))
        processed_image = image.transpose((2, 0, 1))
        processed_image = processed_image.reshape(1, *processed_image.shape)
        
        ### TODO: Start asynchronous inference for specified request ###
        net_input = {'image_tensor': processed_image,'image_info': processed_image.shape[1:]}
        duration_report = None
        inf_start = time.time()
        inference_network.exec_net(net_input, request_id)
        
        ### TODO: Wait for the result ###
        if inference_network.wait() == 0:
            det_time = time.time() - inf_start
            ### TODO: Get the results of the inference request 
            net_output = inference_network.get_output()
            ### TODO: Extract any desired stats from the results 
            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
         
            person_detected = 0
            probs = net_output[0, 0, :, 2]
            for i, p in enumerate(probs):
                if p > prob_threshold:
                    
                    person_detected += 1
                    box = net_output[0, 0, i, 3:]
                    p1 = (int(box[0] * frame_width), int(box[1] * frame_height))
                    p2 = (int(box[2] * frame_width), int(box[3] * frame_height))
                    frame = cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
           
            
            if person_detected != counter:
                last_count = counter
                counter = person_detected
                if current_duration >= 3:
                    duration_prev = current_duration
                    current_duration = 0
                else:
                    current_duration = duration_prev + current_duration
                    duration_prev = 0  # unknown, not needed in this case
            else:
                current_duration += 1
                if current_duration >= 3: #To compat with the video counter +/-1
                    count_publish = counter
                    if current_duration == 3 and counter > last_count:
                        counter_total += counter - last_count
                    elif current_duration == 3 and counter < last_count:
                        duration_report = int((duration_prev / 10.0) * 1000)
            n_person_message = " This is person number {} on the frame".format(counter_total)
            cv2.putText(frame, n_person_message, (40,40),cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            ### TODO: Calculate and send relevant information on ###
            ### person_detected, counter_total and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client.publish('person',
                           payload=json.dumps({
                               'count': count_publish, 'total': counter_total}),qos=0, retain=False)
            if duration_report is not None:
                client.publish('person/duration',payload=json.dumps({'duration': duration_report}),qos=0, retain=False)
                
                if key_pressed == 27:
                    break
 

        ### TODO: Send the frame to the FFMPEG server ###
        #  Resize the frame
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        #print(frame.shape)

    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    #print("Starting mains")
    args = build_argparser().parse_args()
    
    #print(args)
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
