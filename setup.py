from setuptools import setup, find_packages

setup(
  name = 'shapecompiler',
  packages = find_packages(),
  include_package_data = True,
  version = '1.0.0',
  license='MIT',
  description = 'ShapeCompiler-Pytorch',
  author = 'Tiange Luo',
  author_email = 'tiange.cs@gmail.com',
  url = 'https://github.com/tiangeluo/ShapeCompiler',
  install_requires=[
    'axial_positional_embedding',
    'einops>=0.3.2',
    'ftfy',
    'pillow',
    'regex',
    'rotary-embedding-torch',
    'taming-transformers-rom1504',
    'tokenizers',
    'torch>=1.6',
    'torchvision',
    'transformers',
    'tqdm',
    'youtokentome',
    'WebDataset'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
