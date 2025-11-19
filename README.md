# CryoEvap 

This Python software simulates the evaporation of cryogenic liquids in storage tanks. Developed as a project for "HYDROGEN PROCESSING FOR SUSTAINABLE ENERGY", it introduces a modification to the vapor thermal conductivity ($k_v$), implementing a variable model instead of the previously used average value.

### Requirements

* CoolProp >= 6.4.1
* SciPy >= 1.11.3
* Matplotlib >= 3.8.0
* Jupyter >= 1.0.0

### How to use CryoEvap

The workflow of CryoEvap is divided in five steps: module import, tank initialisation, cryogen initialisation, simulation setup and visualisation. The following code snippet illustrates the minimum code necessary to simulate the evaporation of LN2 in a lab-scale storage tank, and it can be used as a template for any scenario.  In step 1, NumPy and Matplotlib are imported to directly visualise the results in the notebook. Additionally, the CryoEvap Tank and Cryogen classes are imported, as they act as an interface for all simulation functionalities. In step 2, the geometrical and heat transfer properties of the tank are defined, as well as the operating pressure and initial liquid filling. In step 3, the cryogen is first initialised with its name and then their properties at the operating pressure are set. In step 4, the grid spacing is set to calculate the number of nodes in the vertical direction, z, and then this value is set as a tank property. The overall roof heat transfer coefficient is also set to illustrate that tank properties can be modified after the tank is constructed. The time_interval property establishes the time-step at which simulation data will be recorded for post-processing. The plot_interval property defines the interval at which vapour temperatures will be plotted. Step 4 ends with the function evaporate, which receives the simulation time and performs the simulation. Finally, Step 5 illustrates the syntax to produce the figures that summarise the results. The parameters t_unit and unit allow the user to control the units of time and the corresponding dependent variables to improve visualisations. Further examples that set up the scenarios and perform the simulations can be found in the Jupyter Notebooks located in the /notebooks folder of CryoEvap GitHub repository.

### What's new in this version

This update replaces the constant $k_{v,avg}$ approximation in CryoEvap with a $z$-dependent variable, requiring modifications to the governing equation for the time step ($dT$) and the boundary conditions that previously relied on average values. A new parameter, $\beta$, was introduced to replace $\alpha$, as the variable model no longer utilizes average $k_{v,avg}$. Furthermore, a switch was implemented in the function mid_tank.set_HeatTransProps(..., k_V_avg=False), allowing users to toggle between the average approximation and the variable model by simply setting the boolean flag, facilitating a direct comparison between the two approaches.

As a first conclusion, there is no difference between both models, which can be observed in the Methane_mid.ipynb file. This verifies that the model assumptions are congruent with what was expected.This confirms the limited contribution of vapor to the BOG.

### For more details and test de code go to Methane_mid.ipynb in -> test.

#### In the code (class tank.py):

Previous Implementation update dT (Constant $k_{v,avg}$):
```python
dT[1:-1] = alpha*d2T_dz2 - (v_z-v_int) * dT_dz + (alpha/self.cryogen.k_V_avg) * S_wall
```
Updated Implementation (Variable $k_v(z)$):
```python
dT[1:-1] = Beta * (k_nuevo[1:-1] * d2T_dz2) - (v_z-v_int)* dT_dz + Beta* (dk_dz * dT_dz) + Beta * S_wall 
```
And the compute the difference for dk_dz:
```python
dk_dz = (k_nuevo[1:-1] - k_nuevo[:-2]) / dz
```

Previous alpha:
```python
alpha = self.cryogen.k_V_avg/(self.cryogen.rho_V_avg*self.cryogen.cp_V_avg) 
```
Updated "alpha" rename to Beta:
```python
Beta = 1/(self.cryogen.rho_V_avg*self.cryogen.cp_V_avg)
```

Updated the temperature shift logic so the $+1e^{-3}$ K offset is now applied globally to the whole T_V array (instead of just the first node) to avoid CoolProp convergence errors. The code:
```python
#Shift temperature 1e-3 K to avoid CoolProp non convergence
        T_V_shift = np.copy(T_V)
        T_V_shift = T_V_shift + 1e-3
```

Robin BC initial condition changes instead k_V average - self.k_V and the correct value for the BC

Changes in def Q_VL where k_V updated:

The new version:
```python
    def Q_VL(self, T_V):
        '''
        Calculate vapour to liquid heat transfer rate
        using the Fourier's law
        '''

        # Temperature gradient at the interface
        dz = (self.z_grid[1] - self.z_grid[0])*self.l_V
        dTdz_i = (-3 * T_V[0] + 4 * T_V[1] - T_V[2])/(2*dz)
        
        return self.k_V[0] * self.A_T * dTdz_i
```
The old version:
```python
    def Q_VL(self, T_V):
        '''
        Calculate vapour to liquid heat transfer rate
        using the Fourier's law
        '''

        # Temperature gradient at the interface
        dz = (self.z_grid[1] - self.z_grid[0])*self.l_V
        dTdz_i = (-3 * T_V[0] + 4 * T_V[1] - T_V[2])/(2*dz)
        
        return self.cryogen.k_V_avg * self.A_T * dTdz_i
```
Changes in def _reconstruct(self), updated k_V:
```python
Q_VL.append(self.k_V[0] * self.A_T * dTdz_i)
```
#### In the code (class cryogen.py):
```python
self.k_V_var = k_V #is defined to store all the vector of $k_V$ values.

self.k_V_var = k_V #is define in the  def init - """Constructor"""
```
#### For the creation of the switch (class tank.py):

Creation of a new property: This property was defined to ensure that the dT equation remains valid/compatible for both scenarios. 1 the constant average $k_V$ and 2 the variable $k_V$.
```python
    # New property for the switch creation, k_V average or k_V variable
@property
def k_V(self):
    if self.k_V_avg:
        "If K_v True takes k_V average"
        return np.ones(len(self.z_grid)) * self.cryogen.k_V_avg
    else:
        return self.cryogen.k_V_var
```
New entrance for the function:
```python
def set_HeatTransProps(self, U_L, U_V, T_air, Q_b_fixed=None, Q_roof=0, eta_w = 0, k_V_avg = True):
```
If true, uses k_V_avg. If false, uses the all vector k_V


```python
# Step 1: Third-party module imports
import sys
sys.path.append("..") # Adds higher directory to python modules path.
# Scientific computing
import numpy as np
# Visualisation
import matplotlib.pyplot as plt
# Import the storage tank Class
from cryoevap.storage_tanks import Tank
# Import Cryogen class
from cryoevap.cryogens import Cryogen

# Step 2: Initialise tank object
Q_roof = 0   		# Roof heat ingress / W
d_i = 8 	    	# Internal diameter / m
d_o = 8.4   		# External diameter / m
T_air = 293.15 		# Temperature of the environment K
U_L = 3.73e-3 		# Liquid overall heat transfer coefficient W/m^2/K
U_V = 1		# Vapour overall heat transfer coefficient W/m^2/K. Its a huge value for notice the change on the k_V_avg and k_V_var
Q_b = 100 		    # Heat transfer rate at the bottom / W
V_tank = 2033   	# Tank volume / m^3
LF = 0.50     		# Initial liquid filling / -
P = 101325  		# Tank operating pressure / Pa

mid_tank = Tank(d_i, d_o, V_tank, LF) # Initialize mid-scale tank
mid_tank.set_HeatTransProps(U_L, U_V, T_air, Q_b, Q_roof, eta_w= 0.8,k_V_avg = False)

# Step 3: Initialise cryogen
methane = Cryogen(name = "methane")
methane.set_coolprops(P)
mid_tank.cryogen = methane

# Step 4: Simulation setup

# Calculate initial evaporation rate
print("The initial evaporation rate of " + methane.name + " is %.1f kg/h" % (mid_tank.b_l_dot * 3600)) 
# Estimate transient period duration
print("Transient period = %.3f s " % mid_tank.tau)
# Minimum number of hours to achieve steady state 
tau_h = (np.floor(mid_tank.tau / 3600) + 1)
# Print simulation time of the transient period for short-term storage
print("Simulation time: %.0i h" % tau_h )
# Calculate boil-off rate
BOR = (mid_tank.b_l_dot * 24 * 3600) / (mid_tank.V * mid_tank.LF * mid_tank.cryogen.rho_L)
print("BOR = %.3f %%" % (BOR * 100))
dz = 0.1 # grid spacing / m
n_z = 1 + int(np.round(mid_tank.l_V/dz, 0)) # Number of nodes
mid_tank.z_grid = np.linspace(0, 1, n_z) # Set dimensionless grid
mid_tank.U_roof = 0 # Roof overall heat transfer coefficient W/m^2/K
evap_time = 3600 * tau_h # Define evaporation time / s
mid_tank.time_interval = 60 # Time-step to record data
mid_tank.plot_interval = evap_time/6 # Interval to plot vapour temperature profiles
mid_tank.evaporate(evap_time) # Simulate the evaporation

# Step 5: Visualisation
mid_tank.plot_tv(t_unit='h') # Vapour temperature
# Specify y-axis units as W, and time units to hours
mid_tank.plot_Q(unit = 'W', t_unit = 'h')
mid_tank.plot_V_L(t_unit='h') # Liquid volume
mid_tank.plot_BOG(unit='kg/h', t_unit='h') # Boil-off gas and evaporation rates
mid_tank.plot_tv_BOG(t_unit='min') # Plot average vapour and boil-off gas temperature

plt.show()
```
#### Errors

To quantify the numerical differences between the two models, the simulation is executed two times. First, the code is run with k_V_avg="False" and the resulting data vectors are saved. Then the simulation is repeated using k_V_avg="True". Finally, the discrepancy between the two datas is evaluated by calculating the Root Mean Square Error and the Mean Absolute Error.


```python
#Save data with k_V_avg = 'True'
BOG_avg = np.copy(mid_tank.data['BOG'])
T_v_avg = np.copy(mid_tank.data['Tv_avg'])
T_BOG_avg = np.copy(mid_tank.data['T_BOG'])
```

```python
#Rerun code with k_V_avg = 'False'
BOG_var = np.copy(mid_tank.data['BOG'])
T_v_var = np.copy(mid_tank.data['Tv_avg'])
T_BOG_var = np.copy(mid_tank.data['T_BOG'])
```

```python
#Errors MAE and RMSE
Erqm_BOG = np.sqrt(np.mean((BOG_var - BOG_avg)**2))
Mae_BOG = np.mean(np.abs(BOG_var - BOG_avg))

Erqm_T_v = np.sqrt(np.mean((T_v_var - T_v_avg)**2))
Mae_T_v = np.mean(np.abs(T_v_var - T_v_avg))

Erqm_T_BOG = np.sqrt(np.mean((T_BOG_var - T_BOG_avg)**2))
Mae_T_BOG = np.mean(np.abs(T_BOG_var - T_BOG_avg))
```

```python
#Prints
print(f"Error BOG")
print(f"Root Mean Square Error: {Erqm_BOG:.2e}")
print(f"Mean Absolute Error: {Mae_BOG:.2e}")
print(f"Error Tv")
print(f"Root Mean Square Error: {Erqm_T_v:.2e}")
print(f"Mean Absolute Error: {Mae_T_v:.2e}")
print(f"Error T_BOG")
print(f"Root Mean Square Error: {Erqm_T_BOG:.2e}")
print(f"Mean Absolute Error: {Mae_T_BOG:.2e}")
```