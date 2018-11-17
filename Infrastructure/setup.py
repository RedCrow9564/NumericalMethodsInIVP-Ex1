from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = {
    Extension(
        'circulant_sparse_product',
        ['circulant_sparse_product.pyx'],
        include_dirs=['.'],
        extra_compile_args=['/openmp'],
        extra_link_args=['/openmp'])
}
setup(
    ext_modules=cythonize(extensions)
)
