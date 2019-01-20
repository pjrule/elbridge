from distutils.core import setup
from Cython.Build import cythonize
import platform
import os

os.environ['CFLAGS'] = '-O3 -std=c++11'
if platform.system() == 'Darwin':
    # It may be necessary to specify the use of libc++
    # when building with clang on macOS.
    # see https://stackoverflow.com/a/40031250
    #     https://stackoverflow.com/a/50407611
    os.environ['CFLAGS'] += ' -stdlib=libc++'
setup(
    name='elbridge',
    ext_modules=cythonize(os.path.join('elbridge', 'cgraph', 'cgraph.pyx')),
    packages=['elbridge']
)
