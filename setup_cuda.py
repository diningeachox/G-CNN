from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gcnn_cuda',
    ext_modules=[
        CUDAExtension('gcnn_cuda', [
            'gcnn_cuda.cpp',
            'gcnn_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
