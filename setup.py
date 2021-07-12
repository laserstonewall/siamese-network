import os
import re

from setuptools import setup, find_packages  # type: ignore

with open('README.md', 'r') as fh:
    long_description = fh.read()


def version():
    version_pattern = r"__version__\W*=\W*'([^']+)'"
    src = os.path.join(os.path.dirname(__file__), 'siamese/__init__.py')
    with open(src, 'r') as f:
        (v,) = re.findall(version_pattern, f.read())
    return v


setup(
    name='siamese',
    version=version(),
    description='PyTorch based Siamese network for image similarity.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    packages=['siamese'],
    python_requires='>=3.7',
)
