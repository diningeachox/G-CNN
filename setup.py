from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='gcnn_functions_cpp',
    ext_modules=[
        cpp_extension.CppExtension('gcnn_functions_cpp', ['gcnn_functions.cpp'])
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension})
