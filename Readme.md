#Group-equivariant CNN (G-CNN)

This project is a pytorch implementation of **Group-equivariant Convolutional Neural Networks** by Cohen and Welling[^1].
We create custom G-convolutional layers and G-pooling layers for various noncommutative groups
of interest, namely:

p4 (translations and rotations of 90 degrees)
p4m (translations and rotations of 90 degrees + reflections)

# Requirements
python >= 3.7
numpy
pytorch >= 1.12.1
CUDA 11.3 (CUDA 11.6 doesn't currently work with the GPU kernels)
libtorch

# Optimization using C++ and CUDA
In the command line, use the following commands
    py setup.py install #C++ extensions
    py setup_cuda.py install #CUDA extensions

These will install the corresponding C++/CUDA functions instead of the native pytorch functions used for
forward() and backward() functions in the GCNN models. 
----
[^1]: Taco S. Cohen, Max Welling **Group Equivariant Convolutional Networks** (2016)
