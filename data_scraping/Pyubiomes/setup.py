
long_desc = '''
# Pyubiomes

Vendored local wrapper around the Cubiomes C library used by the scraping utilities.
'''



from setuptools import setup, find_packages, Extension

# IMPORTANT:
# This project includes C sources directly (via `#include "../cubiomes/*.c"`).
# If compiled under older default C modes (e.g. gnu89), implicit function
# declarations can silently truncate 64-bit values and break worldgen results.
# Force a modern C standard and fail fast on implicit declarations.
extra_compile_args = [
    "-std=c99",
    "-D_DEFAULT_SOURCE",
    "-Werror=implicit-function-declaration",
]

setup(name = 'Pyubiomes', version = '0.2.0', description="Python wrapper for the C library Cubiomes", url="https://github.com/4gboframram/Pyubiomes", long_description=long_desc, include_package_data=True,
long_description_content_type='text/markdown',
packages=find_packages(),
ext_modules = [Extension('Pyubiomes.overworld', sources=['./Pyubiomes/wrap.c'], extra_compile_args=extra_compile_args)],
#package_data={'': ['searches.so']}
)