# Group-equivariant CNN (G-CNN)

This project is a pytorch implementation of **Group-equivariant Convolutional Neural Networks** by Cohen and Welling[^1].
We create custom G-convolutional layers and G-pooling layers for various noncommutative groups
of interest, namely:<br>
<br>
p4 (translations and rotations of 90 degrees)<br>
p4m (translations and rotations of 90 degrees + reflections)<br>

There are a lot of similarities with Cohen and Welling's original implementation [here](https://github.com/tscohen/GrouPy). The main differences are that Cohen and Welling used Tensorflow and CuPy for GPU acceleration, whereas I used Pytorch and Libtorch for extensions with CUDA kernels.

# Requirements
python >= 3.7<br>
numpy<br>
pytorch >= 1.12.1<br>
CUDA 11.3 (CUDA 11.6 doesn't currently work with the GPU kernels)<br>
libtorch<br>
scikit-image <br>
scikit-learn <br>
Microsoft Visual Studio (for Windows users) <br>

# Usage
Before running the code it is **highly recommended** to install the C++/CUDA optimization modules. (Otherwise it may be very slow!)

In the command line, use the following commands <br>
`py setup_cpp.py install` (C++ extensions) <br>
`py setup_cuda.py install` (CUDA extensions)

These will install the corresponding C++/CUDA functions instead of the native pytorch functions used for forward() and backward() functions in the GCNN models.

The training code is in `train.py`. There are two models available currently:
- P4CNN: used to train rotated MNIST
- P4AllCNN: used to train (rotated) CIFAR10

To start training, use the command
`py train.py --model=p4cnn` or `py train.py --model=p4allcnn`

# Task List
* [x] Optimization using C++/CUDA
* [ ] Further GPU optimization with shared memory
* [x] Dataloader with rotations and reflections support
* [x] Unit tests
* [x] Training scripts
* [ ] Benchmark results
* [ ] Visualizations
----
[^1]: Taco S. Cohen, Max Welling **Group Equivariant Convolutional Networks** (2016)
