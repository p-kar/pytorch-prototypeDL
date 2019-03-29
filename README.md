# pytorch-tensorflow-prototypeDL
Pytorch & tensorflow implementation for "Deep Learning for Case-Based Reasoning through Prototypes"


To run the tensorflow implementation for MNIST and FashionMNIST dataset: 

python3 CAE_MNIST.py (or) python3 CAE_fashmnist.py 

Modifications to the decoder architecture: 
The decoder layer with conv2d_transpose is replaced with an interpolation layer followed by convolutional layer. This would help in avoiding the checkerboard effect on the images produced by the decoder. 

To run this implementation for MNIST and FashionMNIST dataset: 

python3 CAE_MNIST_inter.py (or) python3 CAE_fashmnist_inter.py
