from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.1.0'
DESCRIPTION = 'DCGAN architecture based on Fas14MNet for Image Generation'

# Setting up
setup(
    name="FasGAN",
    version=VERSION,
    author="Diganta Dutta",
    author_email="diganta.aimlos@gmail.com",
    url="https://github.com/DigantaD/Fas14MNet/tree/main/dcgan",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['torch'],
    keywords=['image generation', 'cnn', 'gan', 'dcgan', 'vanilla loss'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)