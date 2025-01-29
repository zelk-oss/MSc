    #!/usr/bin/env python
# coding: utf-8

# ## Coupled ocean-atmosphere model version

# This model version is a 2-layer channel QG atmosphere truncated at wavenumber 2 coupled, both by friction
# and heat exchange, to a shallow water ocean with 8 modes.
#
# More detail can be found in the articles:
#
# * Vannitsem, S., Demaeyer, J., De Cruz, L., & Ghil, M. (2015). Low-frequency variability and heat
#   transport in a low-order nonlinear coupled ocean–atmosphere model. Physica D: Nonlinear Phenomena, 309, 71-85.
# * De Cruz, L., Demaeyer, J., and Vannitsem, S.: The Modular Arbitrary-Order Ocean-Atmosphere Model:
#   MAOOAM v1.0, Geosci. Model Dev., 9, 2793–2808, 2016.
#


# ## Modules import
import numpy as np
import sys
import time
from multiprocessing import freeze_support, get_start_method
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing the model's modules
from qgs.params.params import QgParams
from qgs.integrators.integrator import RungeKuttaIntegrator
from qgs.functions.tendencies import create_tendencies
# i need to import all the classes for the functions that i want and understand what type of object 
# I'm dealing with. so I can make my analysis on it. i want time series in the same way i have the fourier component s
# time series 
from qgs.diagnostics.streamfunctions import (
    AtmosphericStreamfunctionDiagnostic, # general base class
    LowerLayerAtmosphericStreamfunctionDiagnostic, # psi^3_a 
    UpperLayerAtmosphericStreamfunctionDiagnostic, # psi^1_a 
    MiddleAtmosphericStreamfunctionDiagnostic, # psi_a, baroclinic 
    OceanicStreamfunctionDiagnostic, # general base class 
    OceanicLayerStreamfunctionDiagnostic, #psi_o
)
from qgs.diagnostics.temperatures import (
    AtmosphericTemperatureDiagnostic, #General base class for atmospheric temperature fields diagnostic.
    MiddleAtmosphericTemperatureAnomalyDiagnostic, # baroclinic streamfunction, middle atmospheric temperature anomaly fields at 500hPa
    MiddleAtmosphericTemperatureDiagnostic, # Diagnostic giving the middle atmospheric temperature fields :math:`T_{\\rm a} = T_{{\\rm a}, 0} + \\delta T_{\\rm a}` at 500hPa
    OceanicTemperatureDiagnostic, # General base class for atmospheric temperature fields diagnostic
    OceanicLayerTemperatureAnomalyDiagnostic, # Diagnostic giving the oceanic layer temperature anomaly fields
    OceanicLayerTemperatureDiagnostic, # Diagnostic giving the oceanic layer temperature fields
    MiddleAtmosphericTemperatureMeridionalGradientDiagnostic, # General base class for atmospheric temperature fields meridional gradient diagnostic. 
    AtmosphericTemperatureMeridionalGradientDiagnostic # Diagnostic giving the meridional gradient of the middle atmospheric temperature fields :math:`\\partial_y T_{\\rm a}` at 500hPa
)
from qgs.diagnostics.variables import (
    VariablesDiagnostic, 
    GeopotentialHeightDifferenceDiagnostic #compute and show the geopotential height difference between points of the model's domain
)
from qgs.diagnostics.multi import MultiDiagnostic

from qgs.diagnostics.wind import (
    AtmosphericWindDiagnostic, # General base class for atmospheric wind diagnostic.
    LowerLayerAtmosphericVWindDiagnostic, # Diagnostic giving the lower layer atmospheric V wind fields :math:`\\partial_x \\psi^3_{\\rm a}`.
    LowerLayerAtmosphericUWindDiagnostic, # Diagnostic giving the lower layer atmospheric U wind fields :math:`- \\partial_y \\psi^3_{\\rm a}`.
    MiddleAtmosphericVWindDiagnostic, # Diagnostic giving the middle atmospheric V wind fields :math:`\\partial_x \\psi_{\\rm a}`.
    MiddleAtmosphericUWindDiagnostic, # Diagnostic giving the middle atmospheric U wind fields :math:`- \\partial_y \\psi_{\\rm a}`.
    UpperLayerAtmosphericVWindDiagnostic, # Diagnostic giving the upper layer atmospheric V wind fields :math:`\\partial_x \\psi^1_{\\rm a}`.
    UpperLayerAtmosphericUWindDiagnostic, # Diagnostic giving the upper layer atmospheric U wind fields :math:`- \\partial_y \\psi^1_{\\rm a}`.
    LowerLayerAtmosphericWindIntensityDiagnostic, # Diagnostic giving the lower layer atmospheric horizontal wind intensity fields.
    MiddleAtmosphericWindIntensityDiagnostic, # Diagnostic giving the middle atmospheric horizontal wind intensity fields.
    UpperLayerAtmosphericWindIntensityDiagnostic, # Diagnostic giving the upper layer atmospheric horizontal wind intensity fields.
    MiddleLayerVerticalVelocity, # Diagnostic giving the middle atmospheric layer vertical wind intensity fields.
)

from matplotlib import rc
rc('font',**{'family':'serif','sans-serif':['Times'],'size':12})

# Initializing the random number generator (for reproducibility). -- Disable if needed.
np.random.seed(21217)

if __name__ == "__main__":

    if get_start_method() == "spawn":
        freeze_support()

    print_parameters = True


    def print_progress(p):
        sys.stdout.write('Progress {:.2%} \r'.format(p))
        sys.stdout.flush()


    class Bcolors:
        """to color the instructions in the console"""
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'


    print("\n" + Bcolors.HEADER + Bcolors.BOLD + "Model qgs v0.2.8 (Atmosphere + ocean (MAOOAM) configuration)" + Bcolors.ENDC)
    print(Bcolors.HEADER + "============================================================" + Bcolors.ENDC + "\n")
    print(Bcolors.OKBLUE + "Initialization ..." + Bcolors.ENDC)
    # ## Systems definition

    # General parameters

    # Time parameters
    dt = 0.1
    # Saving the model state n steps
    write_steps = 1000
    # transient time to attractor
    transient_time = 1.e7
    # integration time on the attractor
    integration_time = 1.e7
    # file where to write the output
    filename = "evol_fields.dat"
    T = time.process_time()

    # Setting some model parameters
    # Model parameters instantiation with default specs
    model_parameters = QgParams()
    # Mode truncation at the wavenumber 2 in both x and y spatial coordinate
    model_parameters.set_atmospheric_channel_fourier_modes(2, 2)
    # Mode truncation at the wavenumber 2 in the x and at the
    # wavenumber 4 in the y spatial coordinates for the ocean
    model_parameters.set_oceanic_basin_fourier_modes(2, 4)

    # Setting MAOOAM parameters according to the publication linked above
    model_parameters.set_params({'kd': 0.0290, 'kdp': 0.0290, 'n': 1.5, 'r': 1.e-7,
                                 'h': 136.5, 'd': 1.1e-7})
    model_parameters.atemperature_params.set_params({'eps': 0.7, 'T0': 289.3, 'hlambda': 15.06, })
    model_parameters.gotemperature_params.set_params({'gamma': 5.6e8, 'T0': 301.46})

    model_parameters.atemperature_params.set_insolation(103.3333, 0)
    model_parameters.gotemperature_params.set_insolation(310., 0)

    if print_parameters:
        print("")
        # Printing the model's parameters
        model_parameters.print_params()

    # Creating the tendencies functions
    f, Df = create_tendencies(model_parameters)

    # ## Time integration
    # Defining an integrator
    integrator = RungeKuttaIntegrator()
    integrator.set_func(f)

    # Start on a random initial condition
    ic = np.random.rand(model_parameters.ndim)*0.01
    # Integrate over a transient time to obtain an initial condition on the attractors
    print(Bcolors.OKBLUE + "Starting a transient time integration..." + Bcolors.ENDC)
    ws = 10000
    y = ic
    total_time = 0.
    t_up = ws * dt / integration_time * 100
    while total_time < transient_time:
        integrator.integrate(0., ws * dt, dt, ic=y, write_steps=0)
        t, y = integrator.get_trajectories()
        total_time += t
        if total_time/transient_time * 100 % 0.1 < t_up:
            print_progress(total_time/transient_time)

    # Now integrate to obtain a trajectory on the attractor
    total_time = 0.
    traj = np.insert(y, 0, total_time)
    traj = traj[np.newaxis, ...]
    t_up = write_steps * dt / integration_time * 100

    print(Bcolors.OKBLUE + "Starting the time evolution ..." + Bcolors.ENDC)
    while total_time < integration_time:
        integrator.integrate(0., write_steps * dt, dt, ic=y, write_steps=0)
        t, y = integrator.get_trajectories()
        total_time += t
        ty = np.insert(y, 0, total_time)
        traj = np.concatenate((traj, ty[np.newaxis, ...]))
        if total_time/integration_time*100 % 0.1 < t_up:
            print_progress(total_time/integration_time)

    print(Bcolors.OKGREEN + "Evolution finished, writing to file " + filename + Bcolors.ENDC)

    np.savetxt(filename, traj)

    print(Bcolors.OKGREEN + "Time clock :" + Bcolors.ENDC)
    print(str(time.process_time()-T)+' seconds')
    
    # diagnostics 
    x_step = 3
    y_step = 3 # these steps decide the resolution of the fields

    # time and trajectory to save the diagnostics 
    reference_time = traj[:,0] # 1D array 
    reference_traj = traj[:, 1:].T # shaped (36, n data points)
     
    # streamufunctions 
    psi_3_a = LowerLayerAtmosphericStreamfunctionDiagnostic(model_parameters, x_step, y_step)
    psi_1_a = UpperLayerAtmosphericStreamfunctionDiagnostic(model_parameters, x_step, y_step)
    psi_a = MiddleAtmosphericStreamfunctionDiagnostic(model_parameters, x_step, y_step, geopotential=True)
    psi_o = OceanicLayerStreamfunctionDiagnostic(model_parameters, x_step, y_step)

    # Temperatures
    delta_T_a = MiddleAtmosphericTemperatureAnomalyDiagnostic(model_parameters, x_step, y_step)
    T_a = MiddleAtmosphericTemperatureDiagnostic(model_parameters, x_step, y_step)
    delta_T_o = OceanicLayerTemperatureAnomalyDiagnostic(model_parameters, x_step, y_step)
    T_o = OceanicLayerTemperatureDiagnostic(model_parameters, x_step, y_step)

    # Wind diagnostics
    wind_diagnostics = [
        LowerLayerAtmosphericVWindDiagnostic(model_parameters, x_step, y_step),
        LowerLayerAtmosphericUWindDiagnostic(model_parameters, x_step, y_step),
        MiddleAtmosphericVWindDiagnostic(model_parameters, x_step, y_step),
        MiddleAtmosphericUWindDiagnostic(model_parameters, x_step, y_step),
        UpperLayerAtmosphericVWindDiagnostic(model_parameters, x_step, y_step),
        UpperLayerAtmosphericUWindDiagnostic(model_parameters, x_step, y_step),
        LowerLayerAtmosphericWindIntensityDiagnostic(model_parameters, x_step, y_step),
        MiddleAtmosphericWindIntensityDiagnostic(model_parameters, x_step, y_step),
        UpperLayerAtmosphericWindIntensityDiagnostic(model_parameters, x_step, y_step),
        MiddleLayerVerticalVelocity(model_parameters, x_step, y_step)
    ]

    # Step 1: Set data and retrieve results for the non-wind diagnostics
    psi_3_a.set_data(reference_time, reference_traj)
    psi_1_a.set_data(reference_time, reference_traj)
    psi_a.set_data(reference_time, reference_traj)
    psi_o.set_data(reference_time, reference_traj)
    delta_T_a.set_data(reference_time, reference_traj)
    T_a.set_data(reference_time, reference_traj)
    delta_T_o.set_data(reference_time, reference_traj)
    T_o.set_data(reference_time, reference_traj)

    # Retrieve diagnostic results for the non-wind diagnostics
    psi_3_a_data = psi_3_a.diagnostic 
    psi_1_a_data = psi_1_a.diagnostic 
    psi_a_data = psi_a.diagnostic
    psi_o_data = psi_o.diagnostic
    delta_T_a_data = delta_T_a.diagnostic
    T_a_data = T_a.diagnostic
    delta_T_o_data = delta_T_o.diagnostic
    T_o_data = T_o.diagnostic

    
    # Step 2: Save the non-wind diagnostics data to .npy files
    np.save('psi_3_a_data.npy', psi_3_a_data)
    np.save('psi_1_a_data', psi_1_a_data)
    np.save('psi_a_data.npy', psi_a_data)
    np.save('psi_o_data.npy', psi_o_data)
    np.save('delta_T_a_data.npy', delta_T_a_data)
    np.save('T_a_data.npy', T_a_data)
    np.save('delta_T_o_data.npy', delta_T_o_data)
    np.save('T_o_data.npy', T_o_data)

    # Step 3: Manually process wind diagnostics, set data, and retrieve diagnostic results
    wind_data = {}

    # Set data and retrieve results for each wind diagnostic individually
    for wind_diag in wind_diagnostics:
        wind_diag.set_data(reference_time, reference_traj)
        wind_data[wind_diag.__class__.__name__] = wind_diag.diagnostic

    # Save wind diagnostic data to .npy files
    #for name, data in wind_data.items():
    #    np.save(f'{name}_data.npy', data)

    # Final message indicating that all diagnostics (including wind) were processed and saved
    #print("All diagnostics (including wind) processed and saved.")


    """ 
    visualization of some stuff, but it was just a test, now would have 
    to update all the names 
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Diagnostics Plots", fontsize=16)

    # Subplot 1: Atmospheric Streamfunction
    axes[0, 0].set_title("Atmospheric Streamfunction")
    psi_a.plot(50, ax=axes[0, 0])

    # Subplot 2: Atmospheric Temperature
    axes[0, 1].set_title("Atmospheric Temperature")
    theta_a.plot(50, ax=axes[0, 1])

    # Subplot 3: Oceanic Streamfunction
    axes[1, 0].set_title("Oceanic Streamfunction")
    psi_o.plot(50, ax=axes[1, 0])

    # Subplot 4: Oceanic Temperature
    axes[1, 1].set_title("Oceanic Temperature")
    theta_o.plot(50, ax=axes[1, 1])

    # Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust rect to fit suptitle
    #plt.show()
    """

    # now I need to output the time series in a file each with its title

    # Initialize a dictionary to store all diagnostics data for CSV export
    csv_data = {}  # Time is not included

    # Non-wind diagnostics
    csv_data['LowerLayerAtmosphericStreamfunction'] = psi_3_a_data.flatten()
    csv_data['UpperLayerAtmosphericStreamfunction'] = psi_1_a_data.flatten()
    csv_data['MiddleAtmosphericStreamfunction'] = psi_a_data.flatten()
    csv_data['OceanicLayerStreamfunction'] = psi_o_data.flatten()
    csv_data['MiddleAtmosphericTemperatureAnomaly'] = delta_T_a_data.flatten()
    csv_data['MiddleAtmosphericTemperature'] = T_a_data.flatten()
    csv_data['OceanicLayerTemperatureAnomaly'] = delta_T_o_data.flatten()
    csv_data['OceanicLayerTemperature'] = T_o_data.flatten()

    # Wind diagnostics
    for name, data in wind_data.items():
        # Use wind diagnostic class names for the column names
        csv_data[name] = data.flatten()

    for key, value in csv_data.items():
        print(f"Column: {key}, Length: {len(value)}")

    # Convert dictionary to a pandas DataFrame
    df = pd.DataFrame(csv_data)

    # Output CSV file
    output_filename = "data_1e5points_1000ws/diagnostics_timeseries.csv"
    df.to_csv(output_filename, index=False)

    print(f"Diagnostics time series successfully saved to {output_filename}")
