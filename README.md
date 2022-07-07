# Towards Line-Restricted Dispatchable Feeders using Probabilistic Forecasts for PV-Dominated Low-Voltage Distribution Grids

This repository contains code for the evaluation of the line-restricted dispatchable feeder (LRDF) and the considered benchmarks.

To start the simulation for the LRDF, run 'Simulation_LRDF.py' (please consider TODO).

To start the simulation for the benchmarks, run 'Simulation_benchmarks.py' (please consider TODO).

Please note the following:
- The used data is of the form:
  - quantiles: dict with keys 'power' and 'energy' 
    - 'power': 48x99x192 array ((forecast hour) x (probability of quantile) x (actual hour of week))
    - 'energy': 47x99x8 array ((forecast hour) x (probability of quantile) x (actual day of week))
  - true power: array (actual hour of week)
- The quantities have different names :
  - l: prosumption [kW]
  - p: power output battery [kW]
  - e: state of energy battery [kWh]
  - g: power exchange between LRDF and grid [kW]

