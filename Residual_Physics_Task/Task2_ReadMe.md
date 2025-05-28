# Residual Physics Task Scripts/Settings

## Settings_and_GroundTruth directory:

This contains the settings csv's that define the parameters used for each index of the parameter sweeps in the residual physics task. It is also where the ground truth trajectories are stored once generated.

## Generate_Ground_Truth_Biharmonic_Kuramoto.jl

File to generate the ground truth trajectories for each dynamical regime of the biharmonic Kuramoto system. Uses phase variables, not phase components, such that transformation to x, y components is required before processing with the standard and hybrid reservoirs. The trajectories will be stored as compressed arrow files in the Settings_and_GroundTruth directory. Will generate all four regimes, synchronous, asynchronous, heteroclinic cycles, and self consistent partial frequency, by varying the first harmonic phase shift gamma_1 as per Clusella, P. et al (2016). A minimal model of self-consistent partial synchrony. _New Journal of Physics_. The trajectories are 62001x10 matrices, corresponding to the 10 oscillators and 6200 second length with dt=0.1), generally they are permuted before use in the downstream scripts, but not before plotting. Run from the command line in the top level directory via "julia ./Residual_Physics_Task/Generate_Ground_Truth_Biharmonic_Kuramoto.jl".

## Task2_ParameterSweep.jl

Designed to be run using a SLURM job-array script, this runs an evaluation for a particular parameter at a particular index of its sweep. It takes command line arguments to define the sweep index (1-20, 1-10 for the regularisation parameter) , the parameter sweep name (i.e. 'SpectralRadius', see settings csv file titles for appropriate names), the ground truth case, (1, 2, 3, 4 for synchronous, asynchronous, heteroclinic cycles, and partial synchrony respectively), and the model type ("ODE", "Standard", and "Hybrid"). Example command line call is: "julia ./Residual_Physics_task/Task2_ParameterSweep.jl 5 ReservoirSize 2 Standard" which would test the standard reservoir using parameters corresponding to the 5th index of the Reservoir Size sweep on the asynchronous regime. Unlike the first task, you should run the ODE model for all sweeps as the plotting script does not account for the fact that the ODE results should not change with reservoir specific parameters. The script will instantiate 40 standard kuramoto model ODE parameter sets (with parameter error), or 40 reservoirs depending on the model type, and train and test each over the 20 test segments of the ground truth trajectories for each of the three variants of the particular regime. It will output csvs containing the valid time of each ODE/reservoir's prediction for each test segment in a 20x40 array. There is an option, 'save_trajectories', to decide whether to output compressed arrow files containing all predicted trajectories for further analysis (this will take up a lot of space for all of the sweeps - around 300GB). Output data will be stored in directories specific to the parameter sweep within the Residual_Physics_Task directory.

To note: the regularisation settings file is indexed in reverse order, this is accounted for in the plotting script, but not in the trajectory script. 

Exact results may vary slightly due to the use of a default random seed when generating the multiplicative errors for the ODE's and hybrid reservoir expert model's parameters.

## Task2_GridSearch.jl

This script is identical to the parameter sweep script, except for the use of the GridSearch settings, effectively acting as another parameter name akin to SpectralRadius or InputScaling in terms of its function in the script. The GridSearch settings csv has entries corresponding to the 8 parameter sets explored, (Figure 7 - grid search cube). As such, the sweep index setting should only be set to 1-8. Example command line call: "julia ./Residual_Phyiscs_Task/Task2_GridSearch.jl 3 3 Hybrid" corresponding to testing the hybrid resevoir on the heteroclinic cycles regime with the 3rd parameter set in the grid search. The script is able to test the ODE model, and the asynchronous regime (number 2), but these were not tested in the paper.

Exact results may vary slightly due to the use of a default random seed when generating the multiplicative errors for the ODE's and hybrid reservoir expert model's parameters.

## Plot_Task2_ParameterSweep.jl

This script will plot Figure 9, the grid of 16 subplots, 1 for each parameter/regime combination across the 4 regimes and 4 parameters considered. As in the parameter error sweep plotting script, you can adjust the number of tests and ODE/reservoir instantiations if you ran a reduced sweep. Modification to plot single sweeps or regimes may be more difficult due to the fact the code was written to create the single figure with all the data at once. The main loop runs across regimes, and then parameters in a subloop, adding plots to a vector 'plot_vector', so if you have a reduced set of results, or only want to plot individual parameter sweeps, then extracting the relevant plots from this vector and plotting them would be the best approach. Missing sweep data will likely cause errors as the script will try to obtain the data, adding exceptions to the loops, or changing the variables they loop through may avoid this. Each plot contains the ODE, Standard and Hybrid results, but this can be modified by changing the 'models' vector on line 41, (and the colours vector) appropriately. 

To note: the regularisation settings file is indexed in reverse order, this is accounted for in the plotting script, but not in the trajectory script. 

Exact results may vary slightly, if using newly generated data, due to the use of a default random seed when generating the multiplicative errors for the ODE's and hybrid reservoir expert model's parameters.

## Plot_Task2_GridSearch.jl

Creates Figure 10, the per reservoir performance (mean valid time over the 20 tests) across Hybrid and Standard, across each parameter set in the grid search, for each regime. It will not plot the illustrative cubes along the top that indicate the position on the grid search cube, as these were added manually to the figure. Due to how the data is loaded, and then used to plot, it may be more difficult to adjust the number of reservoirs, parameter sets, or tests, as this would change the indexing of the variables used to store them. The main data loading loop, line 13, that populates the vector 'all_data' will mean over tests so this may not be an issue, but the subsequent splitting into results specific to each regime would need to be adjusted accordingly, see the comments describing the data structure. As well as plotting and saving the main figure, the script will also print out the maximum and minimum mean valid times for each regime for the standard and hybrid reservoirs to allow comparison of the extremes.

## Plot_Task2_Trajectories.jl

The script follows the format of the trajectory plotting script for the parameter error task. Example call "julia ./Residual_Physics_Task/Plot_Task2_Trajectories.jl 4 Regularisation 4 16 32" to plot the ODE, Standard, and Hybrid x component trajectories for the 4th index of the regularisation sweep on the partial synchrony regime for the 16th test and 32nd reservoir/ODE instance. This will plot trajectories from the grid search runs, but must be adjusted to not read ODE, or asynchronous data as this was not present (would be fine if you have run these yourself). I.e. don't use 2 as the ground truth command line argument (ARG[3]), and remove "ODE" from the model vector in the main plotting loop (line 87).

To note: the regularisation settings file is indexed in reverse order, this is accounted for in the plotting script, but not in the trajectory script. 
