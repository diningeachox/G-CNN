# Group-equivariant CNN (G-CNN)

This project is a pytorch implementation of **Group Equivariant Convolutional Neural Networks** by Cohen and Welling[^1].
We create custom G-convolutional layers and G-pooling layers for various noncommutative groups
of interest, namely:

p4 (translations and rotations of 90 degrees)
p4m (translations and rotations of 90 degrees + reflections)


TODO: Optimization using C++ and CUDA

----
[^1]: *Taco S. Cohen, Max Welling* **Group Equivariant Convolutional Networks** (2016)
