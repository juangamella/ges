from distutils.core import setup

setup(
    name='ges',
    version='0.0.1',
    author='Juan L Gamella',
    author_email='juangamella@gmail.com',
    packages=['ges', 'ges.test', 'ges.scores'],
    scripts=[],
    url='http://pypi.python.org/pypi/ges/',
    license='LICENSE.txt',
    description='Python implementation of the GES algorithm for causal discovery',
    long_description=open('README.txt').read(),
    install_requires=['numpy>=1.15.0']
)
