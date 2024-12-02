from setuptools import setup
from version import __version__

setup(
    name="evpv",          
    version=__version__,              
    install_requires=[],        
    description="A modelling tool to calculate the spatio-temporal charging needs of electric vehicles and the potential for solar-based charging",
    long_description="README.md",
    long_description_content_type="text/markdown",
    author="Jérémy Dumoulin",
    author_email="jeremy.dumoulin@epfl.ch",
    license="MIT",              
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    entry_points={
        'console_scripts': [
            'evpv=evpv.evpv_cli:main',  
        ],
    }    
)