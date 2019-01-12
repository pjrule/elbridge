from distutils.core import setup
from Cython.Build import cythonize
import os

# https://stackoverflow.com/a/40031250
os.environ["CFLAGS"] = "-O3 -std=c++11 -stdlib=libc++"
setup(
    name='elbridge',
    ext_modules=cythonize(os.path.join("elbridge", "mapgraph.pyx")),
    packages=['elbridge']
)
