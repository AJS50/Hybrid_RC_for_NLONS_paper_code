# Supplementary figure related scripts/plotting.

## Multistep_Reservoir_test.jl

This script defines a new ESN struct 'ESN_Multi' that adds a parameter to define how many internal updates to make per external step of the reservoir. It otherwise is a copy of the 'ESN' struct in 'HybridRCforNLONS.jl'. Versions of the standard initialisation, training, and prediction functions are added as appropriate to run the multiple internal steps. 

DynamicalSystems.jl is used to generate the ground truth data used in this script, using new definitions of the biharmonic kuramoto model created to be compatible with it. It will compute a set of valid times across 30 reservoir instantiations across 10 values of internal step count, on a single test trajectory that is disjoint from the training data. Valid times are calculated for three different reservoir parameter sets as described in supplementary figure S16 when predicting the Heteroclinic cycles regime, as well as for the a single parameter set on the Lorenz system. These results are saved in csv's in the Supplementary_Figures directory. 

## Plot_Multistep_Reservoirs.jl

Reads the csv's and plots Supplementary figure S16. 

## Space_Time_Separation_Calc_and_Plot.jl

Reads the ground truth trajectories for each of the four regimes in the Settings_and_GroundTruth directory. Computes the pairwise time separation and Euclidean distance between every 300th point in the trajectory up to 200 points (out of 62001 points, dt=0.1 s) and stores them in two 200x200 arrays. Plots them as per supplementary figure S17.