# **Behavioral Cloning** 

## Writeup 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/hist_old.png "Histogram 1"
[image2]: ./images/hist_new.png "Histogram 2"
[image3]: ./images/data.png "Training Data"
[image4]: ./images/model1.png "Model 1"
[image5]: ./images/model2.png "Model 2"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Content

My project includes the following files:
* train.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model1.h5 and model2.h5 contain trained convolution neural networks 
* writeup.md summarizes the results

#### 2. Usage
Using the Udacity provided [simulator](https://github.com/udacity/self-driving-car-sim) and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model1.h5
```

### Model Architecture and Training Strategy

#### 1. Model Architecture

The model used in this project replicates the [NVIDIA CNN](https://arxiv.org/abs/1604.07316), which was already succesfully used to teach real life cars (https://devblogs.nvidia.com/deep-learning-self-driving-cars/). Input for the model are 320x160 YUV camera images of the front view of the car. They are cropped by 60 pixels from the top and 20 pixels from the bottom (Cropping layer), and normalized to lie between -0.5 and 0.5 (Lambda layer). Dropout and batch normalization layers were added to speed up training and avoid overfitting:

| Layer         |  Activation |  Output Shape          | Parameters  |
| ------------- |:-----:|:--------------:| -----------:|
| Cropping      | - | 80,320,3 | 0 |
| Lambda       | - | 80,320,3      |   0 |
| Conv2D 5x5, 2 strides |ReLU| 38,158,24      |    1824 |
| Batch Normalization | - | 38,158,24      |    96 |
| Conv2D 5x5, 2 strides |ReLU| 17,77,36      |    21636 |
| Batch Normalization | - | 17,77,36      |    144 |
| Conv2D 5x5, 2 strides |ReLU| 7,37,48      |    43248 |
| Dropout 0.2 |-| 7,37,48      |    0 |
| Conv2D 3x3, 1 strides |ReLU| 5,35,64      |    27712 |
| Conv2D 3x3, 1 strides |ReLU| 3,33,64      |    36928 |
| Dropout 0.2 |-| 3,33,64      |    0 |
| Flatten |-| 6336      |    0 |
| Batch Normalization | - | 6336     |    25334 |
| Dense | - | 100     |    633700 |
| Dropout 0.2 |-| 100      |    0 |
| Dense | - | 50     |    5050 |
| Dense | - | 10     |    510 |
| Dense | - | 1     |    11 |

Total Parameters: 796203 (783411 trainable)
(train.py lines 105-124)

#### 2. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (train.py line 125). Additional Keras ModelCheckpoints were used in order to only keep models whose validation loss decreased.

#### 3. Training Data

The model was first only trained on the data set provided by Udacity (24108 RGB images, i.e. 8036 scenes shown from the left, center, right camera), converted to YUV images. However, before feeding the data to the model, some preprocessing had to be done since the histogram of the data looks rather unbalanced:

![Histogram 1][image1]

As we can see, the data heavily favors straight driving which could lead to undesired behavior of the vehicle on curved tracks. There are several ways to tackle this issue. One way would be to randomly drop samples of low steering angles to even out the data. However, this effectively causes a loss of information. I rather chose to oversample the data with high steering angles to give it extra weight in the training. Steering data with angles above 0.02 were sampled 5 times, angles above 0.2 were sampled 20 times and angles above 0.3 were sampled 40 times as much as straight driving data.
This produces are more natural histogram:

![Histogram 2][image2]

As in the NVIDIA use case, left and right camera images were used to augment the training data and to simulate driving close to the left and right lane lines. Steering angles for those images were adjusted by +-0.25 in order to force the car back to the center of the road.
Finally, all images and steering angles were flipped to even out the inherent left-bias of driving around the track counter-clockwise.

The data was split into 0.9 training data and 0.1 validation data sets.

Additionally, after the first few epochs, the data set was expanded by some more recordings of critical sections (e.g. the section right after the bridge with the dirt road leading off the main road) and general recovery data (car driving from the lane lines back to the center of the road). The complete training data can be seen here (55716 images, 18572 scenes):

[![Training Data][image3]](https://www.youtube.com/watch?v=r-06U3oZJT8)

#### 4. Results

After initially training only on the provided data for 10 epochs we already get a quite decent result for the driving behavior of the car (model1.h5):

[![Model 1][image4]](https://www.youtube.com/watch?v=QdZUaFdkQiA)

The car completes the lap while staying on the drivable part of the road and only occasionally crossing the lane markings. At 1:28 it looks like the car interprets the street fencing as the lane line instead of the actual yellow lane markings. At 2:44 it looks like the car is trying to avoid the shadow despite not being a convertible.

I then tried to further train the model with the expanded data set (model2.h5):

[![Model 2][image5]](https://www.youtube.com/watch?v=0ISQxJv6g5Q)

The results are rather mixed. The car still completes the lap (even at higher speed) but produces many more driving mistakes. However, it seems to be able to self-correct even quite big steering mistakes (see 1:26).

Even further training actually diminishes the performance of the model (probably due to overfitting). A better way to improve the model would be to have another look at the training data. One could try to record smoother center line driving or find other ways to adjust the training data (for example shifts or perspective transforms of the training images).