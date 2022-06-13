# Group-equivariant CNN (G-CNN)

This project is a pytorch implementation of **Group Equivariant Convolutional Neural Networks** by Cohen and Welling[^1].
We create custom G-convolutional layers and G-pooling layers for various noncommutative groups
of interest, namely:

p4 (translations and rotations of 90 degrees)
p4m (translations and rotations of 90 degrees + reflections)

The G-equivariant convolutional layer and G-pooling layer is in **gconv.py**, using these custom layers one can build a deep neural network model in very much the same way as a conventional CNN. 

## Benefits of group equivariance 

One of the main reasons that CNNs are so effective is because they can do inference on data up to translation. For instance, in a simple classification task of animals, a CNN can detect the presence of a dog no matter where that dog is positioned in the image. So in this way conventional CNNs are already group-equivariant networks, with the group in question being the translation group of lattices in 2D.

The group-equivariant framework generalizes this construction to larger groups, which include rotations and reflections. In conventional CNNs these symmetries are usually handled by data augmentation, however with group-equivariant CNNs these symmetries are "baked into" the architecture itself. Therefore a p4-equivariant CNN for example, can detect the presence of a dog even if it is not in the standard orientation. This framework can greatly reduce sample complexity as the usual data augmentation used in conventional CNNs is not needed.

Moreover, group-equivariant CNNs are part of an even more general framework known as [geometric machine learning](https://arxiv.org/pdf/2104.13478.pdf) where machine learning is used on non-Euclidean domains. Such domains can exhibit yet more symmetries not found in Euclidean domains, e.g. the 2-sphere is a homogeneous space for the group SO(3) -- the group of 3-dimensional rotations.

**TODO:** Optimization using C++ and CUDA

----
[^1]: *Taco S. Cohen, Max Welling* **Group Equivariant Convolutional Networks** (2016)
