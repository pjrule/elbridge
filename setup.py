import os
import sys
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# see https://dmtn-013.lsst.io/
# see https://github.com/MDAnalysis/mdanalysis/pull/2150/files
compile_args = ['-O3', '-std=c++11']
link_args = []
if sys.platform == 'darwin':
    compile_args += ['-mmacosx-version-min=10.7', '-stdlib=libc++']
    link_args += ['-stdlib=libc++', '-std=c++11']

cgraph_root = os.path.join('elbridge', 'cgraph')

cgraph_module = Extension(
    'elbridge.cgraph',
    sources=[os.path.join(cgraph_root, 'cgraph.pyx')],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    # https://stackoverflow.com/a/14657667
    include_dirs=[numpy.get_include()],
    language='c++')

setup(name='elbridge',
      ext_modules=cythonize(cgraph_module),
      packages=['elbridge'])
