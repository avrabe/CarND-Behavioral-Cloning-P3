[![Code Climate](https://codeclimate.com/github/avrabe/CarND-Behavioral-Cloning-P3/badges/gpa.svg)](https://codeclimate.com/github/avrabe/CarND-Behavioral-Cloning-P3)
[![Build Status](https://travis-ci.org/avrabe/CarND-Behavioral-Cloning-P3.svg?branch=master)](https://travis-ci.org/avrabe/CarND-Behavioral-Cloning-P3)
# CarND-Behavioral-Cloning-P3

_The README thoroughly discusses the approach
taken for deriving and designing a model architecture
fit for solving the given problem._


## The model
I've derived my model from the NVIDIA model describe in [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316).
The image size of the input was adjusted according to the NVIDIA model.
In addition a normalization layer was added directly after the input layer.
Instead of Relu I used [PReLU](https://keras.io/layers/advanced-activations/) which in
addition also contains a learning parameter.
After flattening I've added also several Dropout layer
to combat overcommitment. The overall picture can be seen below.

![Model](model.png)

The default optimizer is Adam and the loss function the mean square error.
It was chosen after testing several optimizer and loss functions.



_The README provides sufficient details of the
characteristics and qualities of the architecture,
such as the type of model used, the number of layers,
the size of each layer. Visualizations emphasizing
particular qualities of the architecture are
encouraged._


_The README describes how the model was trained and
what the characteristics of the dataset are.
Information such as how the dataset was generated and
examples of images from the dataset should be included._

To train the model the data set Udacity [provided](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) was mainly used.

Mainly the center images are used. In addition the left and right images
(when available) is used. For the left and right image an additional adjustment
is added. Afterwards the distribution looks like:
[![Distribution](distribution.png)](distribution.png)
To enhance the effect of the left and right steering images, the amount
of images with a steering degree of 0 is reduced. For the training
the distributed then looks like:
[![Distribution](distribution-filtered.png)](distribution-filtered.png)

The data afterwards is split into a train and validation set.
