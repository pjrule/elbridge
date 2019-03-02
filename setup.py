from distutils.core import setup, Extension
from Cython.Build import cythonize
import os
import sys
import numpy

compile_args = ['-g', '-O3', '-std=c++11', '-stdlib=libc++']
link_args = []
if sys.platform == 'darwin':
    compile_args.append('-mmacosx-version-min=10.7')
    link_args.append('-stdlib=libc++')
    link_args.append('-std=c++11')
cgraph_root = os.path.join('elbridge', 'cgraph')

# see https://dmtn-013.lsst.io/
# see https://github.com/MDAnalysis/mdanalysis/pull/2150/files
cgraph_module = Extension('elbridge.cgraph',
                          sources=[os.path.join(cgraph_root, 'cgraph.pyx')],
                          extra_compile_args=compile_args,
                          extra_link_args=link_args,
                          # https://stackoverflow.com/a/14657667
                          include_dirs=[numpy.get_include()],
                          language='c++')

setup(
    name='elbridge',
    ext_modules=cythonize(cgraph_module),
    packages=['elbridge']
)
