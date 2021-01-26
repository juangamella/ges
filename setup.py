from distutils.core import setup

setup(
    name='ges',
    version='1.0.0',
    author='Juan L Gamella',
    author_email='juangamella@gmail.com',
    packages=['ges', 'ges.test', 'ges.scores'],
    scripts=[],
    url='http://https://github.com/juangamella/ges',
    license='LICENSE.txt',
    description='Python implementation of the GES algorithm for causal discovery',
    long_description=open('README_pypi.md').read(),
    long_description_content_type="text/markdown",
    install_requires=['numpy>=1.15.0']
)
