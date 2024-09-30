# EV-PV
**The EV-PV (Electric Vehicles - Photovoltaics) model is an open-source Python tool designed to calculate the charging needs of privately-owned electric vehicles (EVs), the potential for photovoltaic power generation (PV), and possible synergies between the two (self-sufficiency, self-consumption, ...) in a specific area. The tool primarily focuses on estimating the charging needs for daily commuting. It calculates the mobility demand for this purpose endogenously by combining georeferenced data with transport demand modeling.**

Authors = Jeremy Dumoulin, Alejandro Pena-Bello, Noémie Jeannin, Nicolas Wyrsch

Lead institution = EPFL PV-LAB, Switzerland

Contact = jeremy.dumoulin@epfl.ch 

Langage = python 3 

> :bulb: This `README.md` provides a quick start guide for basic usage of the EV-PV model. Comprehensive documentation for detailed and advanced usage will soon be available on a [Read the Docs](https://readthedocs.org/) page. In the meantime, you can refer to this file and explore the `/examples` folder, which includes a variety of examples, from basic to advanced use cases.

## Table of Contents

1. [Overview of the Model](#overview-of-the-model)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Features](#features)
6. [Contributing](#contributing)
7. [Scientific Publications](#scientific-publications)
8. [Acknowledgment](#acknowledgment)
9. [License](#license)

## Overview of the model

The EV-PV model has three main objectives and corresponding outputs (as shown in the following figure, which illustrates the model’s key inputs, outputs, and processing steps):
1. **Estimate the spatial and temporal charging demand for electric vehicles.** This is done by first calculating the mobility demand within the area of interest. To do so, the later is divided into sub-zones (zoning), and the passenger flows between these zones are estimated, focusing specifically on trips between home and the destinations people travel to for their daily commute (e.g., workplaces, park-and-ride facilities, universities, ...). Based on this, the user can input scenario parameters related to the electric vehicle fleet and the charging behavior to estimate the charging needs.
2. **Calculate the PV power production potential for the area of interest.** This can be done for different configurations (rooftop, ground-mounted, etc.) and relies on the PVLib toolbox. The output primarily includes an hourly capacity factor over a year, along with other standard PV metrics (performance ratio, production in W/m², etc.).
2. **Analyze potential synergies between EVs and PV energy.** This involves evaluating various indicators by combining the PV capacity factor with the EV charging curve for a given PV capacity.

<center>
	<img src="docs/model_overview.png" width="100%"> 
	<p><font size="-1">EV-PV Model overview. Note that many optional input parameters and additionnal outputs (e.g., mobility demand outputs such as daily distance travelled or total passenger-km) are not shown.</font></p>
</center>

## Installation

### Requirements
- **Python**: Ensure Python is installed on your system. 
- **Conda** (optional, but recommended): Use Conda for managing Python environments and dependencies. 
- **Open Route Service API key** (optional, but recommended to perform realistic road-based distance estimation): Sign up for an API key at [OpenRouteService](https://openrouteservice.org/sign-up/).

> :bulb: If you are new to python and conda environments, we recommand installing python and conda via the [Miniconda](https://docs.conda.io/en/latest/miniconda.html) distribution. During the installation, make sure to select "Add Miniconda to PATH" for ease of use.

> :thumbsdown: If you do not want to use conda, we strongly recommend using an other virtual environment manager (venv, ...). However, you can also manually install all the python dependencies (not recommended) using the list of required modules in the `environment.yml` file (Note that the code was developed and tested using python 3.12, so other python version might not work).

### Installation with conda
1. Clone the latest version of the code on GitHub on your local machine. If you are not familiar with git, you can also manually download the folder from GitHub and then run the code. However, you won't be able to contribute to the project.
```bash
$ git clone https://github.com/jeremydumoulin/evpv.git
```

2. Open an Anaconda prompt and create a new conda environment with the required dependencies. 
```bash
$ conda env create -f environment.yml -n your_environment_name
```

3. Activate the conda environment (assuming the environment is named `your_environment_name`). 
```bash
$ conda activate your_environment_name
```

## Project structure
```bash
├───environment.yml
├───LICENSE.md
├───README.md
├───doc/
├───evpv/
│   ├───chargingscenario.py
│   ├───evcalculator.py
│   ├───evpvsynergies.py
│   ├───helpers.py
│   ├───mobilitysim.py
│   └───pvcalculator.py
├───examples/
│   ├───input/
│   ├───output/
│   ├───00_basic_usage.py
│   └───...
└───scripts/
```  
### Available Modules
In the `evpv/` folder, you will find the following modules:

- **Core Classes** (required for any basic usage):
  - **EVCalculator**: Simulates the EV charging demand.
  - **PVCalculator**: Simulates the PV production potential.
  - **EVPVSynergies**: Calculates EV-PV synergy metrics based on the results from the other two classes.

- **Additional Classes** :
  - MobilitySim: Provides mobility demand simulations (for advanced usage).
  - ChargingScenario: Allows simulation of different charging scenarios (for advanced usage).
  - helpers.py: Contains various functions used internally by the other classes (internal use).

### Examples
In the `examples/` folder, you will find various examples illustrating basic and more advanced use cases. We recommend looking at the various scripts, starting with the more basic ones.

### Scripts
In the `scripts/` folder, you will find additionnal helpful scripts, notably a script to fetch georeferenced workplaces from OpenStreetMap.

## Usage

To run the code, create a new Python script and import the necessary modules from the `evpv/` folder.

### Basic Usage

For basic usage, you only need to import the three core classes:

```python
from evpv.evcalculator import EVCalculator
from evpv.pvcalculator import PVCalculator
from evpv.evpvsynergies import EVPVSynergies
```

**Description of Core Classes:**

- **EVCalculator**: EV charging demand simulation. 
- **PVCalculator**: PV production simulation.
- **EVPVSynergies**: EV-PV synergy metrics.

> :bulb: **Tip:** For a quick start, check the example scripts in the `examples/` folder. We recommend starting with the script `00_basic_usage.py`, which demonstrates how to use the three core classes with a minimal set of input parameters. Additionally, the files `01_evcalculator.py` and `02_pvcalculator.py` illustrate the usage of the `EVCalculator` and `PVCalculator` classes in more detail, including all available parameters (required and optional). There is no specific example file for the `EVPVSynergies` class since its complete usage is illustrated in `00_basic_usage.py`.

### Advanced Usage

For advanced usage, you can also import the `MobilitySim` and `ChargingScenario` classes. These classes provide greater control over the EV charging demand estimation by separating mobility demand simulation from charging scenario simulation. In contrast, the `EVCalculator` class performs both tasks in a single step. This can be particularly useful for running mobility demand simulations independently of the EV demand. For example, you can conduct multiple mobility simulations for different types of trips, such as home-to-work and home-to-study, and then aggregate the results to use them in a `ChargingScenario`.

> :bulb: **Info:** Under the hood, the `EVCalculator` class is simply a wrapper class facilitating the use of the `MobilitySim` and the `ChargingScenario` classes.

```python
from evpv.mobilitysim import MobilitySim
from evpv.chargingscenario import ChargingScenario
```

## Features

### Main features
- **Endogenous estimation of daily mobility demand for home-to-work commuting.** To estimate charging needs in a specific area, it is essential to assess the commuting transport demand — specifically, the flow of vehicles between potential origin points (e.g., homes) and destination points (e.g., workplaces), as well as the road-based distance between them. The model estimates this demand internally by dividing the region of interest into traffic zones (based on user-defined spatial resolution) and applying a spatial interaction model to distribute the flow of people between their homes and workplaces (or other parking locations, such as park-and-ride facilities). A key feature of this model is its integration of the self-calibrated gravity model developed by [Lenormand et al.](https://doi.org/10.1016/j.jtrangeo.2015.12.008), which removes the need for transport data specific to the region. For accurate road distance calculations, the model utilizes OpenRouteService to perform routing when available in the region of interest.

- **Spatial and temporal charging needs**. Based on the previously mentioned mobility demand, the model computes the daily spatial and temporal charging needs for electric vehicles. This is done using a scenario-based approach, where the user specifies the characteristics of the vehicle fleet (including the charging power for each vehicle) and the expected charging behaviors. For the spatial demand, the model calculates it for each traffic zone, based on the expected share of people charging either at home or at work. The temporal demand (charging curve) is estimated at an aggregate level, based on the expected arrival times at these locations. A stochastic approach is used here, where for each vehicle, the arrival time, daily distance and vehicle energy consumption (which determines the charging needs), and charging power are randomly sampled. 

- **Electric vehicles and charger power mix**. Users can specify any number of electric vehicles for their case study. This list of vehicles should be provided as input to the `EVCalculator` class in the following format: `[[vehicle1, share1], [vehicle2, share2], ...]`. Each entry consists of a vehicle type and its market share (betwee 0 and 1). For each vehicle, there is an associated typical mix of charger powers. The `EVCalculator` class provides a set of predefined vehicles with typical specifications (see examples). For more advanced use cases, users can manually define custom vehicles by providing a dictionary of parameters, including fuel consumption, vehicle occupancy, and charger power mix.

```python
my_vehicle_1 = {
    'ev_consumption': 0.183, # Electric vehicle consumption (kWh/km)
    'vehicle_occupancy': 1.4, # Vehicle occupancy (average number of people per vehicle)
    'charger_power': {
          'Origin': [[7, 0.68], [11, 0.3], [22, 0.02]], # Mix of charger power at origin (home)
          'Destination': [[7, 0.68], [11, 0.3], [22, 0.02]] # Mix of charger power at destination (workplaces)
        }
    }
my_vehicle_2 = {
    'ev_consumption': 0.15, # Electric vehicle consumption (kWh/km)
    'vehicle_occupancy': 1.4, # Vehicle occupancy (average number of people per vehicle)
    'charger_power': {
          'Origin': [[3.6, 1]], # Mix of charger power at origin (home)
          'Destination': [[2.1, 0.5], [3.6, 0.5]] # Mix of charger power at destination (workplaces)
        }
    }
# More vehicles can be added similarly
```

- **Smart charging.** Users can specify a portion of vehicles that are "smart charging". These vehicles will adjust their own charging pattern etween their arrival and departure time to smooth the overall charging curve. A simple rule-based algorithm is implemented to manage the smart charging behavior of these vehicles.

- **PV power production.** The code offers a simple method to generate hourly photovoltaic (PV) production and other metrics, such as the capacity factor, for a specific location, year, and type of PV system (e.g., rooftop, ground-mounted, etc.). It is built on the PVLib library, which provides robust tools for simulating and analyzing PV system performance.
 
- **Synergies between EV and PV.** Using the charging curve and the local PV production, various metrics can be calculated to assess the potential of PV to meet the charging needs. These metrics include self-sufficiency potential, self-consumption, Spearman correlation, and more. The analysis can be performed for a specific day or over a longer time period.

- **Segmenting mobility demand by groups of commuters.** If necessary, it is possible to run separate mobility demand simulations for different groups of commuters, then aggregate the results to better reflect real-world commuting behavior. This is especially useful when one portion of the population commutes to one type of destination (e.g., workplaces) while others travel to different locations (like park-and-ride facilities or universities). Note that this requires more advanced usage.

- **Mobility demand for other purposes.** While this model primarily focuses on mobility demand for daily commuting, it may also be beneficial to incorporate mobility demand for other activities, such as shopping and leisure. Currently, this demand is not calculated endogenously within the model. However, users can include an optional parameter (called `km_per_capita_offset`) to account for these additional mobility needs on weekdays.

### Limitations & Caveats

#### Mobility and EV Charging Demand

- **Zoning**: The spatial resolution is constrained by the size of the traffic zones, with no downscaling procedure available for the moment. The accuracy may also be affected when using a self-calibrated gravity model, which has not been validated for zones smaller than 5 km². Additionally, rectangular zoning may not be the optimal choice for transport demand modelling.
  
- **Trip purposes**: The mobility demand assumes direct trips between home and destination (and destination to home) with no intermediate stops. Alsom, other trip purposes than daily commuting cannot be modeled.

- **Routing**: Accurate routing depends on OpenRouteService, which requires an internet connection.

- **Weekdays only**: The model only accounts for mobility and charging on weekdays.

- **Zone attractiveness**: When using the number of workplaces from OpenStreetMap (OSM) to determine the attractiveness of a zone for trip distribution, it does not include the number of jobs per workplace. This may reduce the model’s accuracy in areas where the number of jobs per workplace is not evenly distributed.

- **POI charging**: Charging at points of interest (e.g., shopping malls) is not considered in the model.

- **Charging Curve**: The charging curve is aggregated across all traffic zones and is based on the assumed arrival time . The model assumes a normal distribution for arrival times, which may not fully represent real behavior. We also assume that all vehicles charge every day, which may not be entirely accurate (see [Pareschi et al.](https://doi.org/10.1016/j.apenergy.2020.115318)). Additionally, in reality, drivers may not charge their vehicles immediately upon arrival, as their decision could depend on factors such as varying electricity tariffs throughout the day (see, for instance, the charging habits of French citizen [here](https://www.enedis.fr/presse/mobilite-electrique-enedis-publie-deux-nouveaux-rapports-sur-les-habitudes-de-mobilite-et-de)). This behavior could be incorporated into future work to better reflect how drivers respond to price signals (or other incentives) when deciding when to charge.

#### Photovoltaic simulation

- **Weather Data**: The model relies on weather data from PVGIS, which requires an internet connection.

#### EV-PV synergies

- **One charging curve only**: The model uses a single charging curve, though this may change under a stochastic approach. This could be problematic when dealing with a small number of vehicles.

## Contributing
[To be completed]

### Open tasks

- [ ] Finalize the README file
- [ ] Add more advanced examples
- [ ] Enhance inline code documentation
- [ ] Create a readthedocs
- [ ] Make a contributing guide
- [ ] Write some unit tests 

## Scientific publications
[To be completed]

## Acknowledgment 
This project was supported by the HORIZON [OpenMod4Africa](https://openmod4africa.eu/) project (Grant number 101118123), with funding from the European Union and the State Secretariat for Education, Research and Innovation (SERI) for the Swiss partners. We also gratefully acknowledge the support of OpenMod4Africa partners for their contributions and collaboration.

## License

[MIT](https://choosealicense.com/licenses/mit/)
