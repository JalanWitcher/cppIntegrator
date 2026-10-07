import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# 1. Detect the operating system to apply the correct compiler flags
if sys.platform == "win32":
    # Windows (MSVC) optimization flags
    compiler_args = ["/O2", "/fp:fast"]
else:
    # Linux/macOS (GCC/Clang) optimization flags
    compiler_args = ["-O3", "-ffast-math", "-march=native"]

ext_modules = [
    Pybind11Extension(
        "ode_cpp",                  # The name of the module you will import in Python
        [r"dop853_bind.cpp"],         # The C++ source file
        cxx_std=17,
        extra_compile_args=compiler_args,
    ),
]

setup(
    name="ode_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
