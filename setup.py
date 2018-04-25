from setuptools import setup

setup(name='fasttuning',
      version='0.1',
      description='Find your fasttext hyperparameters quickly and easily. ',
      url='https://github.com/vinzeebreak/fasttext-tuning',
      author='Vincent Houlbrèque',
      author_email='vincenthoulbreque@gmail.com',
      license='MIT',
      packages=['fasttuning'],
      install_requires=[
          'fasttext==0.8.3',
      ],
      zip_safe=False)
