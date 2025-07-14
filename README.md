# Toy Updraft Hail Model


Last Updated:
7/14/2025


This repository contains the toy updraft hail model code used in Spychalla and Kumjian (2025a,b) with example input data. This code calculates hail trajectories in variable 2D updrafts (defined in altitude and time) that are completely defined by single-valued parameters. By changing the parameter values, many updrafts can be simulated to explore the range of hail growth possibilities in this idealized framework. The equations used herein are taken from Spychalla and Kumjian (2025a), where the hail model is described in detail.


The script "grow_hail_master.py" will compute hailstone trajectories from specified parameters, or from parameters sampled randomly from coupled or uniform distributions (see the "SP", "MP", and "EX" Monte Carlo simulations in Spychalla and Kumjian, 2025b). The final size, total time aloft, and whether a hailstone was lofted or hit the ground will be returned corresponding to each hailstone along with the parameters used to define that trajectory. 


The folder "data" contains files that can be used as inputs to the hail model in different sampling methods when loaded into memory as xarray.Datasets: "data/coupled_trajectory_scaling_parameters.nc" as ds_hail, "data/coupled_environmental_parameters.nc" as ds_storm, and "data/all_parameters_example_dataset.nc" for ds_all. 


The file "example_hail_model.ipynb" shows an example of how to run a hail trajectory simulation with each of the sampling methods.


References:
- Spychalla, L. K. and M. R. Kumjian, 2025a: An Analytic Approximation of Vertical Velocity and Liquid Water Content Profiles in Supercell Updrafts and Their Use in a Novel Idealized Hail Model. *Journal of Atmospheric Science*, accepted.
- Spychalla, L. K. and M. R. Kumjian, 2025b: Uncoupling the Impact of Environment and Updraft Quantities on Hail Growth using an Idealized Hail Model. *Journal of Atmospheric Science*, accepted.
