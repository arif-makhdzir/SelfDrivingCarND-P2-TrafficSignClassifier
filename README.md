<h1>Traffic Sign Recognition</h1> 




<h2>Build a Traffic Sign Recognition Project</h2>

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

<h3>Data Set Summary & Exploration</h3>

<b>1. Provide a basic summary of the data set and identify where in your code the summary was done. </b>

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy and csv library to calculate summary statistics of the traffic
signs data set:

* The size of training set is <b>34799</b>
* The size of test set is <b>12630</b>
* The shape of a traffic sign image is <b>(32, 32, 3)</b>
* The number of unique classes/labels in the data set is <b>43</b>

<b>2. Include an exploratory visualization of the dataset and identify where the code is in your code file.</b>

The code for this step is contained in the third code cell of the IPython notebook.  

Firstly, I plotted a sample image of each class in the data set. It is a good idea to know how each traffic sign looks like, especially when we want to find sample images from the web to test our neural network:
<img src="./writeupimages/images_each_class.png" alt="Traffic sign each class" /><br>
Here is a histogram of amount of data in the training set for each class:
<img src="./writeupimages/histogram_train.png" alt="Traffic sign each class" /><br>
Here is a histogram of amount of data in the validation set for each class:
<img src="./writeupimages/histogram_validation.png" alt="Traffic sign each class" /><br>
Here is a histogram of amount of data in the testing set for each class:
<img src="./writeupimages/histogram_test.png" alt="Traffic sign each class" /><br>
By plotting a histogram of number of data point for each class, we can see an imbalance amount of data point between different classes. Max amount of data for the training set is 2010 and min amount of data is 180, this is a huge gap. Class imbalances can cause... [insert lecture] 

Max amount of data for the training set is 2010 and min amount of data is 180, this is a huge gap. Class imbalances can cause... [insert lecture]

and the distribution is very similar for the training set, validation set, and test set. 

Note that the fact that the training set & test set has similar distribution of data per class means that the class imbalance might not have that severe of an effect for tst set accuracy, as the least data class wo't be tested that much.

Look at valiation set accuracy per class, see if it corresponds to the amount of data. [insert result]

With this visualization data point, I decided that I need to augment the data so that it is more balanced.
![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by ...

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 






SOLUTION:

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I did a bit of research on best practices for the preprocessing part, and I stmuble upon [this]. I decided to normalize the data, instead of .... From my own testing normalizaton is better than ..., this concur with JY' finding.

The reason why normalization works better is due to the fact that it achieves a mean of closer to 0 and std of ...:

[table]

A mean of closer to zero will avoid local minima in stochastic gradient descent and put data more within the range of where the activation function will be most effective.

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

By plotting a histogram of number of data point for each class, we can see an imbalance amount of data point between different classes and the distribution is very similar for the training set, validation set, and test set. 

Note that the fact that the training set & test set has similar distribution of data per class means that the class imbalance might not have that severe of an effect for tst set accuracy, as the least data class wo't be tested that much.

Max amount of data for the training set is 2000 and min amount of data is 230, this is a huge gap. Class imbalances can cause... [insert lecture]

Look at valiation set accuracy per class, see if it corresponds to the amount of data. [insert result]

With this visualization data point, I decided that I need to augment the data so that it is more balanced.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Just copy paste lenet architecture change the channels and FCL and add dropout

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Epoch I cut off as learning rate stop climbing and start oscilating. The reason why learning is cut off as it starts to oscilate is because any further learning will only result in overfitting, the optimal model is the one that stops right when learning rate starts to plateau.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

Soln:
My architcture is based on Lenet. The reason I choose this architecture is very simple, it is known to be able to achieve 95%+ accuracy on dataset that is quite similar to the traffic sign; so there was no reason for me to reinvent the whole wheel given the project's requirement of 93%+ on validation set acccuracy.

After changing the network width to suit the traffic sign dataset, I was able to get Lenet up and running quickly. My first result was:

[insert result]

The discrepency of accuracy between the training set and validation set tells me that the model is overfitting. So I know need to add regularization, and I had a choice of L1, L2, or dropout. I decided to try only dropout, as it is the best practice for deep neural network regularization. 

Now the question is where should I add the dropout layer and also now I have a new hyperparameter to tune (keep probability). Result:

[Add table of different dropout places and hyperparameters]

hyperparameters tuning:
...








