from setuptools import Extension, setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="gcnn_functions_cpp",
    ext_modules=[
        cpp_extension.CppExtension(
            "gcnn_functions_cpp", ["optimization/gcnn_functions.cpp"]
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
