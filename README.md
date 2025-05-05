<center>
   <img src="docs/logo.png" width="100%"> 
</center>

# EVPV-simulator
**The EVPV-simulator (Electric Vehicles - PhotoVoltaics Simulator) model is an open-source Python tool designed to calculate the spatio-temporal charging needs of privately-owned electric vehicles (EVs) and the potential for locally installed solar photovoltaics (PV) to meet these needs. The tool is primarily suited for modeling mobility demand in cities on weekdays, for which it enables the endogenous computation of daily mobility demand by combining georeferenced data with state-of-the-art spatial trip distribution models. For PV generation, it relies on the PVLib toolbox and integrates various PV installation archetypes (e.g., rooftop, free-standing PV, etc.)..**

Authors = Jeremy Dumoulin, Alejandro Pena-Bello, Noémie Jeannin, Nicolas Wyrsch

Lead institution = EPFL PV-LAB, Switzerland

Contact = jeremy.dumoulin@epfl.ch 

Langage = python 3 

> :bulb: This `README.md` provides a quick start guide for basic usage of the evpv-simulator model. Comprehensive documentation for detailed and advanced usage will soon be available on a [Read the Docs](https://readthedocs.org/) page. 

## Table of Contents

1. [Overview of the Model](#overview-of-the-model)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Features](#features)
6. [Contributing](#contributing)
7. [Scientific Publications](#scientific-publications)
8. [Acknowledgment](#acknowledgment)
9. [License](#license)

## Overview of the model

The **evpv-simulator** model has three main objectives and corresponding outputs (as shown in the Fig.1 , which illustrates the model’s key inputs, outputs, and processing steps):
1. **Mobility Demand Estimation.** Based on a user-defined region of interest and associated geospatial input data (population density, workplaces, points of interest (POIs), and number of EVs to simulate), the tool divides the region of intereste into traffic zones and assesses the mobility demand for commuting by simulating origin-destination for all EVs. 

2. **Charging Demand Analysis.** Using the mobility demand and basic properties of the EV fleet, the model calculates the spatial and temporal charging needs. Users define the preferred charging locations of EV users (at home, at work, or at POIs), typical arrival times, and the available charging powers at each locations. The output includes zone-level charging demand and load curves, assuming uncoordinated charging as a baseline charging strategy.

3. **EV-PV Complementarity.** Using PVLib and PVGIS weather data, the tool simulates the local hourly PV production over a given year. It then assesses how much of the EV charging demand can be met by solar energy, generating key performance indicators like self-sufficiency or self-consumption potentials.

<center>
	<img src="docs/model_overview_3.png" width="100%"> 
	<p><font size="-1">Fig. 1: evpv-simulator overview. Note that many optional input parameters and additionnal outputs are not shown.</font></p>
</center>

## Installation

### Getting python
Ensure Python is installed on your system. This project was developped with **Python 3.12**. Other versions may not be compatible.

If it is your first time with Python, we recommend installing python via [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Many tutorials are available online to help you with this installation process (see for example [this one](https://www.youtube.com/watch?v=oHHbsMfyNR4)). During the installation, make sure to select "Add Miniconda to PATH".

> :thumbsup: Miniconda includes `conda`, which allows you to create a dedicated python environment for `evpv-simulator`. If not using conda, consider alternative environment managers like `venv`. Manual installation of all dependencies via `environment.yml`
is also possible but not recommended.

### Installation 
1. (Optional) Create a Conda environment with Python 3.12. As stated before, it is not mandatory but recommended to use a dedicated environment. Here an example with conda using an environment named *evpv-env*

```bash
$ conda create --name evpv-env python=3.12
$ conda activate evpv-env
```

2. Install evpv as a python package from the GitHub repository
```bash
$ pip install git+https://github.com/jeremydumoulin/evpv-simulator.git
```

## Usage

After installation, you can run the **EVPV model in command-line mode**. This is ideal for users who are not familiar with Python or who want to quickly conduct a simple case study.

First, create a configuration file by copying an existing example such as the [Addis Ababa config file](https://github.com/jeremydumoulin/evpv-simulator/tree/main/examples/Basic_AddisAbaba_ConfigFile). Update it with your own input values and ensure that all required geospatial input data is available (see the config file and `input/` folder for guidance).
> :bulb: We recommend starting with the Addis Ababa example to get familiar with the process.  
> :bulb: Need help in gathering the needed geospatial inputs? See the next section.

Once your config file is ready, open a terminal, activate your conda environment (optional), and run:
```bash
evpv
```
You’ll be prompted to enter the path to your config file:
```bash
Enter the path to the python configuration file: C:\Users\(...)\config.py
```
> :warning: Use absolute paths in the config file, or start the terminal in the same directory to use relative paths.

Here below an animation showing process for the Addis Ababa example:
![](docs/usage.gif)

### Where can I get the needed Geospatial Data?

To run the model, you will need the following four types of geospatial input data:

- **Region of interest**: A GeoJSON file defining the boundary of your study area. For most administrative regions, you can download this from the [GADM dataset](https://gadm.org/).

- **Residential population**: A `.tif` raster file showing population density, in the WGS84 coordinate system. We recommend using the [GHS-POP dataset](https://human-settlement.emergency.copernicus.eu/download.php?ds=pop) at the lowest available resolution.

- **List of workplaces**: A CSV file with the following columns: `name`, `latitude`, `longitude`, and `weight`. You can generate this manually from local data or automatically extract it from OpenStreetMap using the helper script in `scripts/extract_pois_from_osm.py`.

- **List of POIs (Points of Interest)**: Same format and process as for workplaces. Use the same script but with modified inputs.

### Using evpv classes in python scripts (advanced usage)

Advanced users can write custom Python scripts by importing and interacting with core classes from the `evpv/` module:

```python
from evpv.vehicle import Vehicle
from evpv.vehiclefleet import VehicleFleet
# etc.
```

All the classes are located in the `evpv/` folder, as shown in the project structure:

```bash
├── setup.py
├── README.md
├── evpv/
│   ├── vehicle.py
│   ├── vehiclefleet.py
│   ├── region.py
│   ├── mobilitysimulator.py
│   ├── chargingsimulator.py
│   ├── pvsimulator.py
│   ├── evpvsynergies.py
│   ├── evpv_cli.py
│   └── helpers.py
├── examples/
│   └── Basic_AddisAbaba_ConfigFile/
├── scripts/
│   └── extract_pois_from_osm.py
└── docs/
```

**Core Classes**  
- `Vehicle`: Defines a vehicle type.  
- `VehicleFleet`: Manages EV fleet data.  
- `Region`: Defines geospatial properties.  
- `MobilitySimulator`: Simulates trip generation and allocation.  
- `ChargingSimulator`: Estimates charging demand over time and space.  
- `PVSimulator`: Simulates solar energy production.  
- `EVPVSynergies`: Analyzes EV-PV interaction metrics.

**Utilities**  
- `evpv_cli.py`: Command-line interface (see [Usage](#usage)).  
- `helpers.py`: Internal utility functions.

## Features

### Main features
- **Endogenous estimation of daily mobility demand for home-to-work commuting.** To estimate charging needs in a specific area, it is essential to assess the commuting transport demand — specifically, the flow of vehicles between potential origin points (e.g., homes) and destination points (e.g., workplaces), as well as the road-based distance between them. The model estimates this demand internally by dividing the region of interest into traffic zones (based on user-defined spatial resolution) and applying a spatial interaction model to distribute the flow of people between their homes and workplaces (or other parking locations, such as park-and-ride facilities). A key feature of this model is its integration of the self-calibrated gravity model developed by [Lenormand et al.](https://doi.org/10.1016/j.jtrangeo.2015.12.008), which removes the need for transport data specific to the region. For accurate road distance calculations, the model utilizes OpenRouteService to perform routing when available in the region of interest.

- **Mobility demand for other purposes.** While this model primarily focuses on mobility demand for daily commuting, it may also be beneficial to incorporate mobility demand for other activities, such as shopping and leisure. Currently, this demand is not calculated endogenously within the model. However, users can include an optional parameter (called `km_per_capita_offset`) to account for these additional mobility needs on weekdays.

- **Spatial and temporal charging needs**. Based on the previously mentioned mobility demand, the model computes the daily spatial and temporal charging needs for electric vehicles. This is done using a scenario-based approach, where the user specifies the characteristics of the vehicle fleet (including the charging power for each vehicle) and the expected charging behaviors. For the spatial demand, the model calculates it for each traffic zone, based on the expected share of people charging either at home or at work. The temporal demand (charging curve) is estimated with a stochastic approach, where for each vehicle, several factors are randomly sampled: arrival time, daily travel distance and energy consumption (which determines the daily charging needs), and available charging power. Building on the work of [Pareschi et al.](https://doi.org/10.1016/j.apenergy.2020.115318), our model also introduces randomness in the number of vehicles that decide to charge on a given day. This decision is based on a threshold for the state of charge (SoC), with vehicles choosing to charge if their SoC falls below this threshold. 

- **Flexible configuration of EV fleet and charger power**. Supports diverse setups, including mixed charging power levels for each location, customizable vehicle types with user-defined maximum charging power for each vehicle.

- **Smart charging**: By default, the code simulates uncontrolled ("dumb") charging behavior. However, the data is pre-processed to allow easy post-analysis of smart charging strategies. A "peak shaving" algorithm is already implemented, enabling vehicles to adjust their charging patterns between arrival and departure times to smooth the overall charging demand curve. This smart charging behavior is managed using a simple rule-based algorithm.

- **PV power production.** The code offers a simple method to generate hourly photovoltaic (PV) production and other metrics, such as the capacity factor, for a specific location, year, and type of PV system (e.g., rooftop, ground-mounted, etc.). It is built on the PVLib library, which provides robust tools for simulating and analyzing PV system performance.
 
- **Potential for PV to cover the charging needs.** Using the charging curve and the local PV production, various metrics can be calculated to assess the potential of PV to meet the charging needs. These metrics include self-sufficiency potential, self-consumption, Spearman correlation, and more. The analysis can be performed for a specific day or over a longer time period.

### Limitations & Caveats

#### Mobility and EV Charging Demand

- **Zoning**: The spatial resolution is constrained by the size of the traffic zones, with no downscaling procedure available for the moment. The accuracy may also be affected when using a self-calibrated gravity model, which has not been validated for zones smaller than 5 km². Additionally, rectangular zoning may not be the optimal choice for transport demand modelling.
  
- **Trip purposes**: The mobility demand assumes direct trips between home and destination (and destination to home) with no intermediate stops. Alsom, other trip purposes than daily commuting cannot be modeled.

- **Routing**: Accurate routing depends on OpenRouteService, which requires an internet connection.

- **Weekdays only**: The model only accounts for mobility and charging on weekdays.

- **Zone attractiveness**: When using the number of workplaces from OpenStreetMap (OSM) to determine the attractiveness of a zone for trip distribution, it does not include the number of jobs per workplace. This may reduce the model’s accuracy in areas where the number of jobs per workplace is not evenly distributed.

- **Charging Curve**: The charging curve is calculated for each location and traffic zone, based on the zone-specific travel distance distribution and the assumed arrival time. The model assumes a normal distribution for arrival times, which may not fully represent real behavior. We also assume that all vehicles charge every day, which may not be entirely accurate (see [Pareschi et al.](https://doi.org/10.1016/j.apenergy.2020.115318)). Additionally, in reality, drivers may not charge their vehicles immediately upon arrival, as their decision could depend on factors such as varying electricity tariffs throughout the day (see, for instance, the charging habits of French citizen [here](https://www.enedis.fr/presse/mobilite-electrique-enedis-publie-deux-nouveaux-rapports-sur-les-habitudes-de-mobilite-et-de)). This behavior could be incorporated into a "smart charging" algorithm in future work to better reflect how drivers respond to price signals (or other incentives) when deciding when to charge.

#### Photovoltaic simulation

- **Weather Data**: The model relies on weather data from PVGIS, which requires an internet connection.

#### EV-PV complementarity

- **EV-PV System treated as a closed system**: When evaluating EV-PV synergies, we assume that all PV energy is available exclusively for EV charging. This approach ignores other potential loads and limitations associated with distributed PV systems, such as variations between different charging stations equipped with PV.

## Contributing
[To be completed]

### Open tasks

- [x] Update the README file to the new architecture
- [ ] Add more advanced examples
- [x] Enhance inline code documentation
- [x] Feature: energy demand per vehicle by traffic zone (i.e., use the distance distribution per zone and not the aggregated one). This will allow more advanced analysis, like variations in the average number of charging stations per vehicle because of higher travelled distances.
- [x] Create a CLI for easy usage
- [ ] Create a readthedocs
- [ ] Make a contributing guide
- [ ] Write some unit tests 

## Scientific publications
[1] Jérémy Dumoulin et al. A modeling framework to support the electrification of private transport in African cities: a case study of Addis Ababa. arXiv preprint arXiv:2503.03671, 2025. [https://doi.org/10.48550/arXiv.2503.03671](https://doi.org/10.48550/arXiv.2503.03671). 

## Acknowledgment 
This project was supported by the HORIZON [OpenMod4Africa](https://openmod4africa.eu/) project (Grant number 101118123), with funding from the European Union and the State Secretariat for Education, Research and Innovation (SERI) for the Swiss partners. We also gratefully acknowledge the support of OpenMod4Africa partners for their contributions and collaboration.

## License

[GNU GENERAL PUBLIC LICENSE](https://www.gnu.org/licenses/gpl-3.0.html)
