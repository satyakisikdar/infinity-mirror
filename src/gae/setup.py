from setuptools import setup
from setuptools import find_packages

setup(name='gae',
      version='0.0.1',
      description='Implementation of (Variational) Graph Auto-Encoders in Tensorflow',
      author='Thomas Kipf',
      author_email='thomas.kipf@gmail.com',
      url='https://tkipf.github.io',
      download_url='https://github.com/tkipf/gae',
      license='MIT',
      install_requires=['numpy==1.16.6',
                        'tensorflow==1.15.2',
                        'networkx==1.11',
                        'scikit-learn==0.20.4',
                        'scipy==1.2.2',
                        ],
      extras_require={
          'visualization': ['matplotlib==2.1.1'],
      },
      package_data={'gae': ['README.md']},
      packages=find_packages())
