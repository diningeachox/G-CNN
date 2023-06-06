# Group-equivariant CNN (G-CNN)

This project is a pytorch implementation of **Group-equivariant Convolutional Neural Networks** by Cohen and Welling[^1].
We create custom G-convolutional layers and G-pooling layers for various noncommutative groups
of interest, namely:<br>
<br>
p4 (translations and rotations of 90 degrees)<br>
p4m (translations and rotations of 90 degrees + reflections)<br>

# Requirements
python >= 3.7<br>
numpy<br>
pytorch >= 1.12.1<br>
CUDA 11.3 (CUDA 11.6 doesn't currently work with the GPU kernels)<br>
libtorch<br>

# Usage
Before running the code it is **highly recommended** to install the C++/CUDA optimization modules. (Otherwise it may be very slow!)

In the command line, use the following commands <br>
`py setup.py install` (C++ extensions) <br>
`py setup_cuda.py install` (CUDA extensions)

These will install the corresponding C++/CUDA functions instead of the native pytorch functions used for forward() and backward() functions in the GCNN models.

The main code is in `main.py`, where a P4CNN model is initialized and a sample input is passed through and backpropagated once. To run the code, use: <br>
`py -m main` <br>
To use CUDA accelerated forward() and backward(), use <br>
`py -m main --gpu`

# Task List
* [x] Optimization using C++/CUDA
* [x] Dataloader with rotations and reflections support
* [ ] Unit tests
* [x] Training scripts
* [ ] Benchmark results
* [ ] Visualizations
----
[^1]: Taco S. Cohen, Max Welling **Group Equivariant Convolutional Networks** (2016)
