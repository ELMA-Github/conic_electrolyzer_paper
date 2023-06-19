import pyomo.environ as pyo
import numpy as np
import pandas as pd

################# Model for conic hydrogen production over the whole operating range (15-100%) #################

def model_HYP_SOC(config_dict,El,St,Wind,Demand,Prices):
    
     solver = config_dict['solver'] # Choose solver
     
     m = pyo.ConcreteModel() # Initialize optimization model
     
     #%%  Define sets
     T = np.array([t for t in range(0, Prices['N_t'])]) # Sets with timesteps
     m.T = pyo.Set(initialize=T) 
     
     # Define sets for untis
     U = np.array([m for m in range(0, El['N_u'])]) # Sets with electrolyzer units
     m.U = pyo.Set(initialize=U)
     m.TU = pyo.Set(initialize= m.T * m.U)
     
     # Maximum demand over delivery period (e.g., daily or weekly demand)
     N_rep = Demand['D_period'] # Delivery period in hours for minimum hydorgen demand (e.g., 1 day: 24, 1 week: 168)
     N_d = int(len(T) / N_rep)  # Number of delivery periods in the considered case study
     D = np.array([t for t in range(0, N_d)]) # Set of delivery periods
     m.D = pyo.Set(initialize=D)
     TT = np.reshape(T, (-1, Demand['D_period']))
    
     #%% Define the optimization variables
     
     # Optimization variables for the system
     m.p_u = pyo.Var(m.T, bounds = (0, Wind['W'])) # power sold to the grid in hour t [MW] 
 
     # Optimization variables for the electrolyzer units
     m.p_el_u = pyo.Var(m.TU, bounds = (0, El['El_cap'])) # power input in the electrolyzer unit u in hour t [MW]
     m.h_el_u = pyo.Var(m.TU, bounds = (0, None)) # hydorgen produced in the electrolyzer unit u in hour t  [kg/h]
     
     m.z_on = pyo.Var(m.TU, domain=pyo.Binary) # binary variable for on state
     m.z_off = pyo.Var(m.TU, domain=pyo.Binary) # binary variable for off state
     m.z_sb = pyo.Var(m.TU, domain=pyo.Binary) # binary variable for standby state
     m.y_cold = pyo.Var(m.TU, domain=pyo.Binary) # binary variable for cold startup (off-->on)
     
     m.p_s = pyo.Var(m.TU, bounds = (0, None)) # power for unit u , time t
     
     m.xi = pyo.Var(m.TU,bounds = (0, None)) # Auxiliary variable for each conic region
     
     #%%  Define the objective function
     m.obj_val = pyo.Objective(expr = sum(Prices['pi_e'][t]*m.p_u[t] for t in m.T)+
                               sum(Prices['pi_h']*m.h_el_u[t,u]-El['C_cold']*m.y_cold[t,u]*El['El_cap'] for t in m.T for u in m.U), sense=pyo.maximize)
     
     #%%  Define optimization constraints
     
     def cst_bal(m, t):
         return sum(m.p_el_u[t,u] for u in m.U) + m.p_u[t] - Wind['P_w'][t] == 0
     m.cst_bal = pyo.Constraint(m.T, rule=cst_bal)
     
     # Maximum hydrogen demand (daily or weekly)
    
     def cst_h_d_max(m, d):
         return sum(m.h_el_u[t,u] for t in TT[d] for u in m.U) <=  Demand['H_d_max_period']
     m.cst_h_d_max = pyo.Constraint(m.D, rule=cst_h_d_max)
                                  
     ############# Eletrolyzer constraints ##################
     # Status
     def cst_bin(m,t,u):
         return m.z_on[t,u] + m.z_sb[t,u] + m.z_off[t,u] == 1
     m.cst_bin = pyo.Constraint(m.TU, rule=cst_bin)
     
     def cst_p_el_lb_u(m,t,u):
         return m.p_el_u[t,u]>=El['P_min']*El['El_cap']*m.z_on[t,u]+El['P_sb']*El['El_cap']*m.z_sb[t,u]
     m.cst_p_el_lb_u = pyo.Constraint(m.TU, rule=cst_p_el_lb_u)
     
     def cst_p_el_ub_u(m,t,u):
         return m.p_el_u[t,u]<=El['P_sb']*El['El_cap']*m.z_sb[t,u]+El['El_cap']*m.z_on[t,u]
     m.cst_p_el_ub_u= pyo.Constraint(m.TU, rule=cst_p_el_ub_u)
     
     def cst_cold_start(m,t,u):
         if t == m.T[1]:
             return m.y_cold[t,u]==0
         else:
             return m.y_cold[t,u]>=m.z_off[t-1,u]+m.z_on[t,u]+m.z_sb[t,u]-1
     m.cst_cold_start = pyo.Constraint(m.TU, rule = cst_cold_start)
     
     # Power consumption
    
     def cst_p_u_tot(m,t,u):
         return m.p_el_u[t,u]==m.p_s[t,u]+El['P_sb']*El['El_cap']*m.z_sb[t,u]
     m.cst_p_u_tot = pyo.Constraint(m.TU, rule = cst_p_u_tot)
     
     def cst_ps_lb(m,t,u):
         return m.p_s[t,u]>=m.z_on[t,u]*El['P_min']*El['El_cap']
     m.cst_ps_lb = pyo.Constraint(m.TU, rule = cst_ps_lb)
     
     def cst_ps_ub(m,t,u):
         return m.p_s[t,u]<=m.z_on[t,u]*El['El_cap']
     m.cst_ps_ub= pyo.Constraint(m.TU, rule = cst_ps_ub)
     
     # Hydrogen production
     def cst_h_prod(m, t, u):
            return m.h_el_u[t,u] == El['D_0']*m.z_on[t,u]+El['D_1']*m.p_s[t,u]+El['D_2']*m.xi[t,u]
     m.cst_h_prod = pyo.Constraint(m.TU, rule = cst_h_prod)
     
     def cst_h_prod_q(m,t,u):
         return m.xi[t,u]>=m.p_s[t,u]**2
     m.cst_h_prod_q = pyo.Constraint(m.TU, rule = cst_h_prod_q)
     
     #%% ############# Solve the problem ##################
     # Define solver
     Solver = pyo.SolverFactory(solver)
     if solver == 'gurobi':
        Solver.options['threads'] = config_dict['threads']
        
     SolverResults = Solver.solve(m, tee=True)
     SolverResults.write()

     if config_dict['print_model']==1:
        m.pprint()

     comp_time =  SolverResults.Solver.time

     
     #%% ############# Save results ##################
     obj_val=m.obj_val()
     
     c_el=np.atleast_2d(np.array([Prices['pi_e']])).T
     P_w=np.atleast_2d(np.array([Wind['P_w']])).T
     p_el =  np.atleast_2d(np.array([sum(m.p_el_u[t,u].value for u in m.U) for t in m.T])).T
     p_u = np.atleast_2d(np.array([m.p_u[t].value for t in m.T])).T
     h_el =  np.atleast_2d(np.array([sum(m.h_el_u[t,u].value for u in m.U) for t in m.T])).T
     
     p_el_u = np.reshape(np.array([m.p_el_u[t,u].value for t in m.T for u in m.U]), (Prices['N_t'],El['N_u']))
     h_el_u = np.reshape(np.array([m.h_el_u[t,u].value for t in m.T for u in m.U]), (Prices['N_t'],El['N_u']))
    
     z_on = np.reshape(np.array([m.z_on[t,u].value for t in m.T for u in m.U]), (Prices['N_t'],El['N_u']))
     z_sb = np.reshape(np.array([m.z_sb[t,u].value for t in m.T for u in m.U]), (Prices['N_t'],El['N_u']))
     z_off = np.reshape(np.array([m.z_off[t,u].value for t in m.T for u in m.U]), (Prices['N_t'],El['N_u']))
     y_cold = np.reshape(np.array([m.y_cold[t,u].value for t in m.T for u in m.U]), (Prices['N_t'],El['N_u']))
     p_s = np.reshape(np.array([m.p_s[t,u].value for t in m.T for u in m.U]), (Prices['N_t'],El['N_u']))   
     
     df = pd.DataFrame(np.concatenate((c_el, P_w, p_u, p_el, h_el, p_el_u, h_el_u, z_on, z_sb, z_off, y_cold), axis=1),
                       columns =['C_el','P_w','p_u','p_el','h_el']+['p_el_u']*El['N_u']+['h_el_u']*El['N_u'] + ['z_on']*El['N_u'] + ['z_sb']*El['N_u'] + ['z_off']*El['N_u'] + ['y_cold']*El['N_u'])
    
     xi = np.reshape(np.array([m.xi[t,u].value for t in m.T for u in m.U]), (Prices['N_t'],El['N_u']))
     z_s = None # variable not defined for HYP-L and for HYP-SOC, but only for HYP-MIL and HYP-MISOC
     return  obj_val, comp_time, df, p_el, p_el_u, p_u, h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi
