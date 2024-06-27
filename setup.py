from setuptools import setup, find_packages

setup(
    name='radar',
    version='0.1',
    description='A Python library for learning and teach radar signal processing.',
    author='Blinn',
    author_email='your.email@example.com',
    url='https://github.com/nrb5089/radar/',
    #packages=find_packages(),  # Automatically find packages in the project
	#py_modules=['radar', 'core', 'util'],  # List of modules in the root
	packages=find_packages(include=['radar', 'radar.*']),
    install_requires=[
        'numpy',
        'scipy',
        'opencv-python',
    ],
    python_requires='>=3.10',  # Specify the Python version requirement
)
