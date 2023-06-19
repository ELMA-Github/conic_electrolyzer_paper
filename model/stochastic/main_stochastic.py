import numpy as np
import pandas as pd 

import components, input_data_stoch, run_model_stoch, results_stoch, input_stochastic

# %% Define config
config_dict = {
      'eff_type': 3, # Choose model for hydorgen production curve1 (1:HYP-MIL, 2: HYP-L, 3: HYP-SOC, 4: HYP_MISOC)
      'solver':'gurobi', # choose solver (gurobi)
      'print_model': 0, # control model printing
      'threads':8
      }
# %% Time horizon
N_t=24 # Number of hours (it has to be <= the length of the input dataset)

# %% Number of scenarios
N_omega = 10 # number of scenarios

# %% Input data

################# Input system data #################
# Electrolyzer
El_cap_tot = 1.0 # Total electrolyzer capacity  [MW]
N_u = 1 # Number of unit (integer value)
El_cap = El_cap_tot/N_u # Capacity of each elctrolyzer unit [MW]

# Wind
W_E_ratio = 2 # Wind/Electrolyzer capacity ratio  
W = W_E_ratio*El_cap_tot # Wind installed capacity [MW]

# Hydorgen demand
D_coverage_max = 0.9 # Maximum percentage of demand to be delivered over the day or week (percentage of full load hours)

################# Electrolyzer data #################
El = {'El_cap_tot': El_cap_tot, # Total electrolyzer capacity [MW]
      'El_cap':El_cap, # Electrolyzer unit (i.e., stack) capacity [MW]
      'N_u': N_u, # number of units
      'P_min':0.15, # % minimum load
      'P_sb': 0.01, # % Power consumption in stand-by state
      'C_cold': 50 # Cold startup cost  [EUR/MW]
      } 

# Find power corresponding to max. efficiency
P_eta_max = components.p_eta_max_fun(El) # around 28% of maximum power

# Define segments
num = 2

P_segments = [
    [El['P_min'],1], #1 #DOES NOT WORK FOR LEFT = 0
    [El['P_min'],P_eta_max,1], #2
    [El['P_min'],P_eta_max,0.64, 1], #3
    [El['P_min'],P_eta_max,0.52,	0.76, 1], #4
    [El['P_min'],P_eta_max,0.46,	0.64, 	0.82, 1], #5
    [El["P_min"],P_eta_max,  0.4240,	0.5680,	0.7120,	0.8560,  1], #6
    [], #7
    [], #8
    [], #9
    [El["P_min"],0.215, P_eta_max,  0.37,	0.46,	0.55,	0.64,	0.73,	0.82,	0.91,  1], #10
    [], #11
    [], #12
    [], #13
    [El["P_min"],0.215, P_eta_max,  0.34,	0.40,	0.46,	0.52,	0.58,	0.64,	0.70,	0.76,	0.82,	0.88,	0.94, 1], #14
    [],#15
    [],#16
    [],#17
    [],#18
    [],#19
    [El["P_min"],0.1933, 0.2367, P_eta_max,  0.3224,	0.3647,	0.4071,	0.4494,	0.4918,	0.5341,	0.5765,	0.6188,	0.6612,	0.7035,	0.7459,	0.7882,	0.8306,	0.8729,	0.9153,	0.9576, 1], #20
    [],#21
    [],#22
    [],#23
    [El['P_min'],0.1825,	0.2150,	0.2475, P_eta_max, 0.3160,	0.3520,	0.3880,	0.4240,	0.4600,	0.4960,	0.5320,	0.5680,	0.6040,	0.6400,	0.6760,	0.7120,	0.7480,	0.7840,	0.8200,	0.8560,	0.8920,	0.9280,	0.9640, 1],#24
]


p_val = np.array(P_segments[num-1])*El["El_cap"]

if config_dict["eff_type"]==4: # HYP_MISOC
    p_val = np.array([El['P_min'], 0.35, 1])*El["El_cap"]
El.update({'p_val': p_val })

components.initialize_electrolyzer(El,config_dict) # Initialize electrolyzer (approximation coeff., etc.)
components.plot_el_curves(El, config_dict) # Plot the nonlinear and approximated curve

################ Storage data #################
St = {} # no storage in our case study

################ Wind data #################
Wind = {'W': W, # Wind installed capacity [MW]
        'CP': input_stochastic.data_CP_w(N_t,N_omega), # hourly wind capacity factors
        'CP_omega': input_stochastic.data_CP_scenarios(N_t,N_omega), # hourly  wind capacity factor in each scenario
        'N_omega': N_omega, #number of scenarios
        'pi_omega': [1/N_omega] * N_omega} #probability of each scenario
P_w = Wind['CP']*Wind['W'] # hourly wind production [MWh]    
Wind.update({'P_w': P_w})
P_w_omega = Wind['CP_omega']*Wind['W'] # hourly wind production in each scenario[MWh] 
Wind.update({'P_w_omega': P_w_omega})
################# Prices ################# 
pi_h = 2.1 # Cost of hydrogen [EUR/kg]
pi_e= input_data_stoch.data_pi_e(N_t)
Prices = {'pi_e': pi_e, # Electricity prices [EUR/MWh]
          'N_t':len(pi_e), # Number of hourly timesteps
          'pi_h': pi_h, # Cost of hydrogen [EUR/kg]
          'pi_up' : 0.9*pi_e, # upregulation balancing price [EUR/MWh]
          'pi_down' : 1.2*pi_e # upregulation balancing price [EUR/MWh]
          } 

################# Hydrogen demand #################
H_d_max = El["El_cap_tot"]*El['eta_full_load'] # Maximum hourly demand [kg/h]
D_period = 24 # Delivery period in hours (1 day: 24, 1 week: 168)
if Prices['N_t']<24:
   D_period = Prices['N_t']   
H_d_max_period = H_d_max * D_period * D_coverage_max # Maximum demand to be satisfied over the time period [kg]
Demand = {'H_d_max': H_d_max,
        'D_period': D_period,
        'D_coverage_max': D_coverage_max,
        'H_d_max_period': H_d_max_period}

# %% Solve the optimization problem
obj_val, comp_time, df, p_el, p_el_u, p_u, h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi, p_el_RT, delta_up, delta_down, h_el_RT, p_el_u_RT, h_el_u_RT, z_s_RT, p_s_RT, xi_RT, z_on_RT, z_off_RT, z_sb_RT, y_cold_RT = run_model_stoch.solve_fun(config_dict,El,St,Wind,Demand,Prices)

print("Objective value = ", obj_val, " EUR")
pd.set_option('display.max_columns', None)
print(df.round(4))
print("Computational time = ", comp_time, " s")

# %% Results

# Plot optimal dispatch
results_stoch.plot_results(config_dict,El,St,Wind,Demand,Prices, obj_val, p_u, p_el, h_el, z_on, z_sb, z_off, y_cold, df, z_s, p_s, comp_time)

