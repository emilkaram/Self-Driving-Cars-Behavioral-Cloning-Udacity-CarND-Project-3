# **Behavioral Cloning** 

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
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
*video.mp4 recorded simulator driving in autonomous mode using the trained model.h5
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. 1.	Modified nvidia model architecture has been employed

## The input Data Preprocess:
-read images files and labels(Steering angle)
-add original with the corresponding steering angle measurements to the data frame
-Add the flipped images with the -ve of corresponding steering angle measurements to the data frame to have more data to generalize the model
-added correction offset to the steering angle to compensate for left and right camera positons 
-data normalization and images cropping is be done in the CNN layers
## The Model
My model consists of a convolution neural network 
The input data is normalized in the model using a Keras lambda layer(x:x/255-0.5)

The normalized data images is cropped in the model using a Keras Croping2D layer (cropping=((70,25),(0,0))

The model then got 5 Convolution layers with subsamples(2,2)and activation RELU layers to introduce nonlinearity.

The model then got 5 flatten layers 

The model uses mean square error(MSE) as a loss measure and adam as the optimizer

Here is the model summary:

![final model](https://github.com/emilkaram/Udacity-CarND-Behavioral-Cloning-Project3/blob/master/images/model_arch.png)


#### 2. Attempts to reduce overfitting in the model

I tried the model with 10% and 20% dropout layers in order to reduce overfitting but was doing better performance without dropout so I decide to skip it.
The model was trained and validated on different data sets to ensure that the model was not overfitting (Used different methods for data input keyboard , mouse , my drone drone transmitter frysky i6s connected to my laptop USB port)
Also tried to collect more laps on track1 
Used flipped imaged to generalize the model and not to be biased toward one direction
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track all the time.



#### 3. Model parameter tuning

The model uses mean square error as a loss measure and adam as the optimizer, so the learning rate was not tuned manually  
#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use modified nvidia model
My first step was to use a convolution neural network model similar to the navidia model I thought this model might be appropriate because of the simple architecture and easy to tune
In order to gauge how well the model was working, I split my image and steering angle data into a training 80% and validation set 20%.
My first model loss was too high on test and validation , I loaded more data and preprocessed the data as explained above to generalized the model 
The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. to improve the driving behavior in these cases , I did try different layers parameters and finally try different numbers of epochs , 10, 5, 2, 1 , and epochs =1 or 2 works the best because after 2 epochs the loss start to increase 
Also data shuffling was a key to achieve genialized model with a minimum loss

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network 
Here is a visualization of the architecture 
Input data shape
(47730, 160, 320, 3)
Shuffle and split 80% train set and 20% validation set
train on 38184 samples, validate on 9546 samples

![finalmodel](https://github.com/emilkaram/Udacity-CarND-Behavioral-Cloning-Project3/blob/master/images/model_arch.png)


End Results
Train on 38184 samples, validate on 9546 samples
Epoch 1/1
38184/38184 [==============================] - 865s 23ms/step - loss: 0.0037 - val_loss: 0.0070


#### 3. Creation of the Training Set & Training Process

Input data shape
(47730, 160, 320, 3)
Shuffle and split 80% train set and 20% validation set
train on 38184 samples, validate on 9546 samples



To capture good driving behavior, I recorded two laps on track one using center camera and left and right cameras. Here is an example images:

https://github.com/emilkaram/Udacity-CarND-Behavioral-Cloning-Project3/blob/master/images/center_2018_04_20_21_31_45_856.jpg

https://github.com/emilkaram/Udacity-CarND-Behavioral-Cloning-Project3/blob/master/images/left_2018_04_20_21_39_52_048.jpg

https://github.com/emilkaram/Udacity-CarND-Behavioral-Cloning-Project3/blob/master/images/right_2018_04_20_21_39_54_081.jpg


 

Then I repeated this process on track two in order to get more data points.

To augment the data I also flipped images and angles thinking that this would generalize the model and avoid biase to a cetraten lap direction  

After the collection process, I had 47730 number of data points. I then preprocessed this data as descriped above.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 1 as evidenced by minum loss as shown below I used an adam optimizer so that manually training the learning rate wasn't necessary.

and here is the end results
Train on 38184 samples, validate on 9546 samples
Epoch 1/1
38184/38184 [==============================] - 865s 23ms/step - loss: 0.0037 - val_loss: 0.0070


