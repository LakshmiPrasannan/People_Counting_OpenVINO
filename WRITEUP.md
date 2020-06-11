# Project Write-Up
Here I am trying to explain the end to end pipeline of this project and mention some of my approaches taken in completing this one. 

This project consists of:
1. Intel OpenVINO toolkit's model optimizer and inference engine.
2. Computer Vision techniques for preprocessing of image data.
3. MQTT Server to publish the statistics inferred from the model.
   The Model Optimizer converted the TensorFlow Model to the IR format to handle the input streams and infer.
    The step to conver the model is done outside of the main.py. 
    Model selected: Faster R-CNN Inception V2 COCO
    Tensor Flow tar file: ![faster_rcnn_inception_v2_coco_2018_01_28.tar.gz](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
    Steps to convert the model to IR format:
    * Downloaded the tar file mentioned
    * using tar -xvf - extracted the file
    * Using the following command generated the xml, bin and mapping file for the frozen model
    
    ***python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model /home/workspace/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config /home/workspace/faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json***
    
    Inside the main function the following steps were taken combining with the inference steps specific to Network Model of OpenVINO Toolkit. 
    * Loading the model specifying the xml and bin file where the xml gave the configuration details and bin file gives the weights of the pretrained model.
    * Use CV library functions to capture image frame from the video stream
    * Understading the shape of the data being transferred, here its the size of the image frame being passed
    * Preprocess the frame data using CV techniques like resizing to convert into model compatable format and transposing the matrix representation
    * Start the Asynchronous execution of the inference model by passing the input data and request id. 
    * Wait for the output and if wait is zero proceed with inferring statistics from the output data
    * Print the wait time as inference time using CV methods onto the frame
    * Logic to find the number of persons detected in a frame and the total count of people appeared in the frame
        * Person is detected if the output tensors have probability value greater than a particular threshold
        * If a new person enters the frame, person_detected counter is incremented and the last count of persons in the frame is updated
        * The duration is calculated as the person stays in the frame, i.e. when the probability threshold is greater in the same loop
        * By understanding the Inference time for the model and video's time taken I have chosen 15s value as condition to update the total count as well as the previous duration.
        * If the current duration is found to be greater than the previous duration and the total counted person is not 1 (i.e. the person is not the first person on the frame), an alert message is printed on the frame. 
    * current_count, total_count and duration to the MQTT server, where duration and total_count are published together and current_count is published after every loop. 
    * for the user to quit explicitly key_break is provided
The command to run this program is as below:

***python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm***

## Explaining Custom Layers
![Architecture](https://github.com/LakshmiPrasannan/People_Counting_OpenVINO/blob/master/Faster_R_CNN_Architecture.png)

In the layers of the model of Faster R-CNN object detection model
    1. Proposing feature extraction over the image before region proposal so that the CNN need not be run unnecessarily on every predictions on the image. In some cases we can even have 2000 predictions, to improve the efficiency we can need to run only 1 CNN after feature selection. 
    2. Running one CNN over the entire image so that the machine learns by itself to classify the detection
    3. Using region proposal network (RPN) to propose the best possible region for detection
Here’s how the RPN worked:

* At the last layer of an initial CNN, a 3x3 sliding window moves across the feature map and maps it to a lower dimension (e.g. 256-d)
* For each sliding-window location, it generates multiple possible regions based on k fixed-ratio anchor boxes (default bounding boxes)
* Each region proposal consists of 
    a) an “objectness” score for that region and 
    b) Four coordinates representing the bounding box of the region to make the detection visible to the user. 
    
 4.A softmax layer that outputs the class probabilities directly so that that the result is now more like a classification than a regression. 
    
  In this layers the convolution layers are executed using Batch Normalizations in dimensions 1*1, 3*3 and using Max pooling for striding. 
    

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were running the video with TensorFlow's original documentation code for Object Detection using Faster R-CNN in the ![link](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb)

Though the person detection was happening on the video, there were intermittency in the detection that showed less accuracy and didn't detect the person while the person was moving.
This sometimes caused incorrect numbers being counted in Total count value, thinking the new box obtained is a new person appearing on the screen.
However this issue was completely overcome when the model was converted to IR format and run through inference requests. 

In the OpenVINO Toolkit the MQTT server helped to see the realtime output unlike the TensorFlow code where the output could not be deployed to the web (Which needed extra coding and infrastructure requirements.)


## Assess Model Use Cases

Some of the potential use cases of the people counter app are in 
* Understanding the social distancing checkers now especially in the Covid times. 
* Checking for count of people who have used a certain service like say a public transport. 
* Check for people who spent more time than the average time and help them separately in case of students. 

Each of these use cases would be useful because this helps us in keeping safety norms, counts and statistics that can help for future planning and expansion. 

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:

With some of the models used the model accuracy kept varying with the factors like camera focal length/image. Especially while using lite models. However with the current model this wasn't an issue and we haven't done any conversions to the frame like grey image to understand the scene better. 

## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [MobileNetSSD]
  - ![Model Source](https://github.com/C-Aniruddh/realtime_object_recognition)
  - I converted the model to an Intermediate Representation with the following arguments from the caffe model of the MobileNet SSD which had MobileNetSSD_deploy.caffemodel and MobileNetSSD_deploy.prototxt.txt file. 
  - The model was insufficient for the app because this model doesn't detect person when the person isn't facing camera. 
  - I tried to improve the model for the app by using a model that can detect persons from any angle to the camera. 
  
- Model 2: [Faster R-CNN ResNet 50 COCO]
  - ![Model Source](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments from a tensor flow model with the frozen_inference_graph.pb, pipeline.config and faster_rcnn_support.json files. 
  - The model was insufficient the inference time was too much and the model differed to detect with changes in lighthing, actions of humans. 
  - I tried to improve the model for the app by another TensorFlow model. 

- Model 3: [Faster R-CNN Inception V2 COCO]
  - ![Model Source](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments frozen_inference_graph.pb, pipeline.config and faster_rcnn_support.json files. 
  - The model was sufficient for the app because it correctly detected humans even when not facing camera and not altering much with input variations with a considerably decent inference time. 
  
