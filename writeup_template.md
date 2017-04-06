
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 126) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.  dropout ratio between convolution laters is small (0.3) and between dense layer is larger (0.5)

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, image flipping data augmentation, left and right camera angle adjustment to sterring.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to LeNet I thought this model might be appropriate because it has been quite successful at image classification with multiclass and is in general powerful enough when I was getting poor results, I switched to Nvidia architecture.  I never moved back to LeNet after that.  It might make sense to try again with LeNet now that I have a working solution.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I added drop out and added more training data.

Initially I used train data form both tracks to generalize the model more however it was hard to get a model that handled turns well.  So I just switched to data from track1.  Also lot of train data was casually geerated by manuvering the car using keyboard and not being too carefull.  I found this made final model performance not as stable throughout the track.  So it threw away most of train data and only used high quality train data where I slowly manuvered the car in similater and being careful to keep it in center of lane.  This gave much better model performance.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture  consisted of a convolution neural network very similar to Nvidia architecture published here: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/


####3. Creation of the Training Set & Training Process

Initially I used train data form both tracks to generalize the model more however it was hard to get a model that handled turns well.  So I just switched to data from track1.  Also lot of train data was casually geerated by manuvering the car using keyboard and not being too carefull.  I found this made final model performance not as stable throughout the track.  So it threw away most of train data and only used high quality train data where I slowly manuvered the car in simulater and being careful to keep it in center of lane (three laps on track one).  This gave much better model performance.

My training data finally consisted of three laps on track one center lane driving along with Udacity provided data.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
