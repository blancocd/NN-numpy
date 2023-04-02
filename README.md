# NN-numpy
A simple neural network framework in Numpy.

A multi-layer fully-connected neural network. The net has N inputs of size D, a hidden layer dimension of H, and output dimension C.  We train the network with a mean squared error (MSE) loss function. The network uses a ReLU nonlinearity after each FC layer but the last one. The outputs of the last fully-connected layer are passed through a ReLU. 

This framework lends itself to encoding a picture so that given its coordinate (x,y) in the image, it returns the corresponding rgb value. However, the neural net will prioritize learning general features at lower frequencies so a better approach is to map the (x,y) coordinate through a Gaussian Fourier Feature mapping. Source: https://bmild.github.io/fourfeat/ 
This assignment is inspired by and built off of the authors' demo. 

Starting code and utils.py, viz.py functions provided by CS444@UIUC.