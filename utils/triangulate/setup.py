from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension('PyWrapper',
                sources=['PyWrapper.pyx', './triang_tools/src/triangulation.cc', './triang_tools/src/misc.cc'],
                include_dirs=['./triang_tools/include'],
                extra_compile_args=['-std=c++11']
                )

setup(
    name = "PyWrapper",
    ext_modules = cythonize(ext),
)

#python3.5 setup.py build_ext --inplace
#python setup.py build_ext --inplace
