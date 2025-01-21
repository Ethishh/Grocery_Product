# GrocerEye - A YOLO Model for Grocery Object Detection

## Abstract
I trained a YOLOv3 model, pretrained on ImageNet, on the Frieburg grocery dataset that was annotated with object detection labels. By training for 6000 iterations and 13 hours on a Google Colab GPU, the model was able to achieve 84.59% mAP and 70.10% IOU on the test set. I demonstrate the effectiveness of the model on images from the test set and inference from a natural video. Though the model performs well on the test set, it does not seem to be effective enough to deploy for real time object detection at this time. I discuss the challenges and possible extensions of this work. 


## Related Work
The related works that I consulted in this investigation are listed below with brief descriptions:
1. Background information on food banks and the Covid crisis: https://www.feedingamerica.org/sites/default/files/2020-10/Brief_Local%20Impact_10.2020_0.pdf
2. Paper on classification with Frieburg Dataset: http://ais.informatik.uni-freiburg.de/publications/papers/jund16groceries.pdf
3. Frieburg Dataset annotated with object detection labels: https://github.com/aleksandar-aleksandrov/groceries-object-detection-dataset
4. Information on training with Darknet in Colab: https://medium.com/@quangnhatnguyenle/how-to-train-yolov3-on-google-colab-to-detect-custom-objects-e-g-gun-detection-d3a1ee43eda1
5. Paper on YOLOv3: https://pjreddie.com/media/files/papers/YOLOv3.pdf
6. Article on converting data with Roboflow: https://blog.roboflow.com/how-to-convert-annotations-from-pascal-voc-to-yolo-darknet/
7. pjreddie version of darknet: https://github.com/pjreddie/darknet
8. pjreddie Darknet YOLOv3 tutorial: https://pjreddie.com/darknet/yolo/ 
9. AlexeyAB version of darknet: https://github.com/AlexeyAB/darknet
10. Interpreting IoU: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/


## Methodology
The approach that I used for solving this problem was to use a YOLOv3 model and train on an object detection dataset with images of groceries. 

### The Dataset
The Frieburg dataset contains 4947 total images across 25 classes of common food items that were found in grocery stores (3). Each class has at least 97 images, many with several object instances. Because this dataset was originally used for classification, each image only has one class of object in it. The Frieburg Object Detection dataset that is linked above was originally in Pascal VOC format, so I converted it to YOLOv3 format using Roboflow (6). I also used Roboflow to partition the dataset into training and test sets with an 80/20 split. Roboflow has a different version of YOLOv3 than the pjreddie/darknet version that I used, so I wrote the bash script ```reorganization.sh``` to make the dataset compatible with the pjreddie/darknet version of YOLOv3.

### The Model
Because in practical applications, this network would have to be very fast for use in real time object detection, I chose to use a YOLOv3 model over other popular object detection techniques such as Faster RCNN. Also, due to the objects being large in the images from the Frieburg dataset, I anticipated that the spatial constraints from YOLOv3 that cause detection on small objects to be ineffective would not be a significant concern. The Frieburg dataset is relatively small, I so initialized the network with YOLOv3 weights trained on ImageNet which was provided by Joe Redmon (8). This would hopefully improve our model and decrease our training time. Because I do not have an NVIDIA GPU in my laptop, I trained the network on Google Colab. The colab notebook that I wrote for training is ```GrocerEye_YOLOv3_Darknet.ipynb```, and I adapted code from Quang Nguyen for training in Colab (4). I trained for 6000 iterations, which took around 13 hours on the Google Colab GPU.

## Experiment and Evaluation
I am evaluating my results with mean average precision and intersection over union metrics, and the results are shown below. To do this, I had to use the AlexeyAB fork of darknet, which entailed some manual reorganization of the data (9). You can find the evaluation code in the colab notebook ```GrocerEye_Eval.ipynb```. Also, I evaluated my results qualitatively by inspecting images from inference on the test set and inference on a video that I took with my phone of food items in my house. The inference was done in another Colab notebook called ```GrocerEye_Inference.ipynb```, and I also adapted code from Quang Nguyen for this part as well (4).

## Results
### mAP Analysis
The results from running the AlexeyAB mAP script are shown below.

![alt text](writeup/mAP.JPG?raw=true)

We see that we actually have quite high average precision on most of our classes, with objects such as jam and water having very high average precision. By looking at these objects I see that the packaging styles are fairly uniform across products and brands, making them easy to identify. However, some objects like chocolate and pasta have lower average precision, which may be due to variety in packaging. The mean average precision is 84.59% and the average intersection over union is 70.10%. The mean average precision being a high 84.59% indicates that the model is good at identifying the classes. For reference, YOLOv3-416 has mAP on COCO of 55.3% (8). COCO is a very complex dataset, so this is expected to be lower, but I think that the comparison helps to put our mAP into perspective. An average intersection over union being 70.10% indicates that our model is good at identifying the bounding boxes for objects (10). 

### Qualitative Analysis
The qualitative analysis that I am presenting references the examples below. We see that for the inference examples for the test set, the model actually performs pretty well! The model is able to identify several object correctly such as corn, tea, coffee, vinegar, chocolate, milk, pasta, and beans. The model incorrectly identifies an instance of pasta, soda, and vinegar. In the video inference example, we see that the model doesn't perform as well in a more natural setting. It is able to correctly identify coffee most of the time, and it can identify the tea, beans and vinegar some of the time. The ramen was not in the dataset, but I thought it might be close enough to pasta for the model to identify (but this was evidently not the case). The model is able to identify the object instances, but the model has trouble reliably classifying them. For instance, the model thinks that the tea and ramen are candy a lot of the time, perhaps because of the packaging and color. The model also thinks that the vinegar is either juice or water sometimes, perhaps because of the shape of the bottle.

## Examples
The following examples are from the inference Colab notebook. These are 10 randomly sampled images from the test set and used for inference. Above each are the classes for the objects.
![alt text](writeup/predictions.png?raw=true)
