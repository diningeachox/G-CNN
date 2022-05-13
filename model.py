from gconv import GConv2d, GMaxPool2d

'''
In this file we implement G-equivariant versions of some well-known
neural network architectures such as LeNet and ResNet
'''

class GResnet(nn.Module):
