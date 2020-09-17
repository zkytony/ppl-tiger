from setuptools import setup

setup(name='ppl-tiger',
      packages=['ppl_tiger'],
      version='0.1',
      description='Spatial Frame-of-reference OOPOMDP',
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'pomdp_py==1.2.0',
          'opencv-python',  # for some tests
          'docutils',
          'pyro-ppl',
          'pandas',
          'scikit-learn',
          'sklearn',
          'torch',
      ])
