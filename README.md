# EVPV
**EVPV is ...**

Authors = Jeremy Dumoulin, Alejandro Pena-Bello, Noémie Jeannin, Nicolas Wyrsch

Lead institution = EPFL PV-LAB, Switzerland

Contact = jeremy.dumoulin@epfl.ch 

Langage = python 3 

> :bulb: This `README.md` provides a quick start guide for basic usage of the project. Comprehensive documentation for detailed and advanced usage will soon be available on a [Read the Docs](https://readthedocs.org/) page. In the meantime, you can refer to this file and explore the `/examples` folder, which includes a variety of examples, from basic to advanced use cases.

## Installation

## Project structure

## Usage

## Features

## Contributing

## Scientific publications

## Acknowledgment 

## Good to know

## License


# GTFS4EV
**GTFS4EV (GTFS for Electric Vehicles) is a python code simulating the daily electric energy demand of a transport system powered by electric vehicles using GTFS data. It also comes wit helper functions and scripts to estimate the main environmental (exposure to air pollution, reduction in CO2 emissions) and economic benefits (savings for owner-operators) of vehicle electrification.** 

GTFS data is used to model the operation of a the vehicles along each trip according to the stop_times.txt and shapes.txt files. Then, the frequencies.txt file is used to estimate the number of vehicles in operation on each trip. Estimated electrical power dissipation and energy consumption are based on a user-specified energy demand per kilometer (kWh/km). 

More generally, the code provides the following features:

- GTFS data consistency and filtering
- Assessment of general transit indicators based on the GTFS data: area (km2) covered by the routes, average trip length, distance between stops, etc.
- Simulation of the transport system operation and associated indicators: number of vehicles in operation, speed, distance travelled, etc.
- Power and energy profile of the electric vehicle fleet
- Helper functions to calculate topological characteristics (under development) and exposure to traffic-related air pollution 

> :memo: **Note:** GTFS stands for General Transit Feed Specification. It is a standardized data format designed for scheduled transport systems, in the form of a set of .txt files describing various aspects of the system and linked together using a relational data structure. Importantly, it contains both spatial and temporal data, paving the way for mobility simulation. The main useful files and their links are shown in the following figure. Note that for the code to work, some GTFS files that are officially optionnal are required, such as shapes.txt and frequencies.txt. For more detailed information about GTFS data, please visit the [google documentation](https://developers.google.com/transit/gtfs).

<center>
	<img src="doc/gtfs_data_structure.png" width="600"> 
	<p><font size="-1">GTFS data structure showing the relations between the different tables. Adapted from: J. Vieira, Transp. Res. Part D, 2023.</font></p>
</center>

authors = Jeremy Dumoulin, Alejandro Pena-Bello, Noémie Jeannin, Nicolas Wyrsch

institution = EPFL PV-LAB, Switzerland

contact = jeremy.dumoulin@epfl.ch 

langage = python 3  

## Project structure
```bash
│   .gitattributes  
│   .gitignore  
│   .env  
│   environment.yml  
│   LICENSE.md  
│   README.md  
│     
├───doc  
│       gtfs_data_structure.png  
│         
├───gtfs4ev  
│   │   gtfsfeed.py  
│   │   helpers.py  
│   │   topology.py  
│   │   trafficsim.py  
│   │   tripsim.py  
│   │   __init__.py  
│
├───scripts
├───input  
└───output
```  

## Quickstart Guide to Retrieve the Modeling Results of the Paper

1. Open the folder containing the code on your local machine.

2. Setup conda environment: 
Open a terminal and create a conda environment with necessary dependencies.
```bash
$ conda env create -f environment.yml
$ conda activate gtfs4ev
```
> :thumbsdown: If you do not want to use a virtual environment, you can also manually install the python dependencies (not recommended). The code was developed and tested using python 3.12, and the list of required modules is available in the `environment.yml` file.

3. Configure environment variables:
Set up environment variables (INPUT_PATH and OUTPUT_PATH) in the .env file.

4. Run Simulation:
Execute `scripts/run_all.py` to run the simulation for all studied cities. Depending on your system and settings, this process may take several hours. Refer to the documentation below for customization options or to create your own scripts using the gtfs4ev modules.

5. View Results:
After completion, review the results in the `/output` folder. Key outputs include aggregated data in `_InputSummary.csv` and `_OutputSummary.csv`, along with city-specific outputs like air pollution exposure maps.

## More general usage the gtfs4ev code

### Environment variables
Open the `.env` file and set the paths of the various environment variables to the desired paths:

- INPUT_PATH: Path to the input files
- OUTPUT_PATH: Path to store the output files

### Input and output data
The **input data** should be placed in the dedicated `/input` folder. The input files are of two types:   

- GTFS datasets  
- GIS data (such as OSM roads)

> :warning: <span style="color:#dd8828">**Important:** If your GTFS data is in the form of a .zip file, you'll need to extract it manually into the `/input` folder.</span>

After running the simulation, some **output data** will be automatically stored in the dedicated `/output` folder. We also recommend using this folder to store any other simulation results.

> :memo: **Note:** If you would like to use other folders to store input and output data, you can set your own path by changing the values of the OUTPUT_PATH and INPUT_PATH variables defined in the `.env` file. This file also contains some other usefull variables, such a physical constants. 

### Running simulations

To run the code, you need to import the various modules you want to use from the `gtfs4ev/` package into a new python script. 

> :bulb: **Tip:** For a quick start, we recommend looking at the `run_all.py` file. This files come with different sections of code that you can simply uncomment to get to familiar with the code step by step. Note that the folder also contains a `preprocess_gtfs.py` file, which is a script containing pre-processing rules for some GTFS data.

You will find two types of modules in the package:
* The core **classes** of gtfs4ev: GTFSFeed, TripSim, TrafficSim
* Helper functions: **helpers.py** and **topology.py** (under development)

Both can be imported as modules to be used in the python script.
```python
# Core classes
from gtfs4ev.gtfsfeed import GTFSFeed
from gtfs4ev.tripsim import TripSim
from gtfs4ev.trafficsim import TrafficSim

# Optionnally, helpers
from gtfs4ev import helpers as hlp
```

The helper modules speak for themselves. The following is a brief description of the core classes, but the user is referred to the classes themselves for a more detailed description:

* **GTFSFeed**: Holds the GTFS feed. Is instantiated using a GTFS data folder in the input folder. Provides features for checking GTFS data, filtering data (e.g. to keep only services present on certain days), and extracting general information about the feed. This class is purely about analyzing and curating data; no modeling involved here.  
* **TripSim**: Simulates the behaviour of a single vehicle or a vehicle fleet along a trip and extracts relevant metrics. Is instantiated using a GTFSFeed, the trip_id, and the electric vehicle consumption (kWh/km). Provides both operational metrics and power/energy profiles.
* **TrafficSim**: Simulates the behaviour of a vehicle fleet along a set of several trips. Is instanciated using a GTFSFeed, a list of trip_ids, and a list of corresponding electric vehicle consimption (kWh/km). Provides operational metrics and profiles for the set of trips. 

> :bulb: **Tip:** For a quick start, we recommend running the main.py file, which comes with different sections of code that you can simply uncomment to get to familiar with the code step by step.

## License

[MIT](https://choosealicense.com/licenses/mit/)
