from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


# The Cython modules to setup
ext_modules = [
    Extension('circulant_sparse_product',
              ['circulant_sparse_product.pyx'],
              include_dirs=['.'],
              extra_compile_args=['/fopenmp'],
              extra_link_args=['/fopenmp'])
]

# Run the setup command
setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)