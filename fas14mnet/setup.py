from setuptools import setup, find_packages

classifiers = [
	'Development Status :: 5 - Production/Stable',
	'License :: OSI Approved :: Apache Software License',
	'Programming Language :: Python :: 3'
]

setup(
	name = 'Fas14MNet',
	version = '0.0.2',
	description = 'CNN architecture for Image Classification',
	long_description = open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
	url = 'https://github.com/DigantaD/Fas14MNet/tree/main/fas14mnet',
	author = 'Diganta Dutta',
	author_email = 'diganta.aimlos@gmail.com',
	license = 'Apache 2.0',
	classifiers = classifiers,
	packages = find_packages(),
	install_requires = ['torch'],		
)