#Group-equivariant CNN (G-CNN)

This project is a pytorch implementation of **Group-equivariant Convolutional Neural Networks** by Cohen and Welling[^1].
We create custom G-convolutional layers and G-pooling layers for various noncommutative groups
of interest, namely:

p4 (translations and rotations of 90 degrees)
p4m (translations and rotations of 90 degrees + reflections)

**TODO:** Optimization using C++ and CUDA

----
[^1]: *Micikevicius, P.; Narang, S.; Alben, J.; Diamos, G.; Elsen, E.; Garcia, D.; Ginsburg, B.; Houston, M.; Kuchaiev, O.; Venkatesh, G. & Wu, H.* **Mixed Precision Training** (2017)
