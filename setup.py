from setuptools import setup
from version import __version__

setup(
    name="evpv",          
    version=__version__,                  
    description="A modelling tool to calculate the spatio-temporal charging needs of electric vehicles and the potential for solar-based charging",
    long_description="README.md",
    long_description_content_type="text/markdown",
    author="Jérémy Dumoulin",
    author_email="jeremy.dumoulin@epfl.ch",       
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    install_requires=[
        "folium==0.14.0",
        "geopandas==0.14.2",
        "geopy==2.4.1",
        "numpy==2.2.5",
        "openrouteservice==2.3.3",
        "pandas==2.2.2",
        "pvlib==0.11.0",
        "pyproj==3.7.1",
        "rasterio==1.3.10",
        "scipy==1.15.2",
        "shapely==2.1.0",
        "timezonefinder==6.5.3",
    ],
    entry_points={
        'console_scripts': [
            'evpv=evpv.evpv_cli:main',  
        ],
    }    
)