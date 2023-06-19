import pyomo.environ as pyo
import numpy as np
import pandas as pd

################# Model for piecewise linear hydrogen production#################
def model_HYP_L(config_dict,El,St,Wind,Demand,Prices):

    solver = config_dict['solver'] # Choose solver
    
    m = pyo.ConcreteModel() # Initialize optimization model
    
    #%%  Define sets
    T = np.array([t for t in range(0, Prices['N_t'])]) # Sets with timesteps
    S = np.array([t for t in range(0, El['N_s'])]) # Sets with linearization segments
    m.T = pyo.Set(initialize=T) 
    m.S = pyo.Set(initialize=S) 
    m.TS = pyo.Set(initialize= m.T * m.S)
    
    U = np.array([m for m in range(0, El['N_u'])]) # Sets with electrolyzer units
    m.U = pyo.Set(initialize=U)
    m.TU = pyo.Set(initialize= m.T * m.U)
    m.TUS = pyo.Set(initialize= m.T*m.U*m.S)
    
    # Maximum demand over delivery period (e.g., daily or weekly demand)
    N_rep = Demand['D_period'] # Delivery period in hours for minimum hydorgen demand (e.g., 1 day: 24, 1 week: 168)
    N_d = int(len(T) / N_rep)  # Number of delivery periods in the considered case study
    D = np.array([t for t in range(0, N_d)]) # Set of delivery periods
    m.D = pyo.Set(initialize=D)
    TT = np.reshape(T, (-1, Demand['D_period']))
    
    # Sets with scenarios for RT
    O = np.array([t for t in range(0, Wind['N_omega'])]) # Sets with scenarios
    m.O = pyo.Set(initialize=O) 
    m.TO  = pyo.Set(initialize= m.T * m.O)
    m.TSO = pyo.Set(initialize= m.T*m.S*m.O)
    
    m.TUO = pyo.Set(initialize= m.T*m.U*m.O)
    m.TUSO = pyo.Set(initialize= m.T*m.U*m.S*m.O)
    m.DO = pyo.Set(initialize= m.D*m.O)
    
    #%% Define the DA optimization variables
    
    # Optimization variables for the system
    m.p_u = pyo.Var(m.T, bounds = (0, Wind['W'])) # power sold to the grid in hour t [MW]

    # Optimization variables for the electrolyzer units
    m.p_el_u = pyo.Var(m.TU, bounds = (0, El['El_cap'])) # power input in the electrolyzer unit u in hour t [MW]
    m.h_el_u = pyo.Var(m.TU, bounds = (0, None)) # hydorgen produced in the electrolyzer unit u in hour t  [kg/h]
    
    m.z_on = pyo.Var(m.TU, domain=pyo.Binary) # binary variable for on state
    m.z_off = pyo.Var(m.TU, domain=pyo.Binary) # binary variable for off state
    m.z_sb = pyo.Var(m.TU, domain=pyo.Binary) # binary variable for standby state
    m.y_cold = pyo.Var(m.TU, domain=pyo.Binary) # binary variable for cold startup (off-->on)
    
    m.p_s = pyo.Var(m.TU, bounds = (0, None))
    
    #%% Define the RT optimization variables (per scenario)
    m.delta_up = pyo.Var(m.TO, bounds = (0, None)) # power up to the grid in hour t and scenario omega [MWh]
    m.delta_down = pyo.Var(m.TO, bounds = (0, None)) # power down to the grid in hour t and scenario omega [MWh]

    # Optimization variables for the electrolyzer units
    m.p_el_u_RT = pyo.Var(m.TUO, bounds = (0, El['El_cap'])) # power input in the electrolyzer unit u in hour t [MW]
    m.h_el_u_RT = pyo.Var(m.TUO, bounds = (0, None)) # hydorgen produced in the electrolyzer unit u in hour t  [kg/h]
    
    m.p_s_RT = pyo.Var(m.TUO, bounds = (0, None))
    
    m.z_on_RT = pyo.Var(m.TUO, domain=pyo.Binary) # binary variable for on state
    m.z_off_RT = pyo.Var(m.TUO, domain=pyo.Binary) # binary variable for off state
    m.z_sb_RT = pyo.Var(m.TUO, domain=pyo.Binary) # binary variable for standby state
    m.y_cold_RT = pyo.Var(m.TUO, domain=pyo.Binary) # binary variable for cold startup (off-->on)
    
    #%%  Define the objective function

    m.obj_val = pyo.Objective(expr = sum(Prices['pi_e'][t]*m.p_u[t] for t in m.T)+
                              sum(Prices['pi_h']*m.h_el_u[t,u]-El['C_cold']*m.y_cold[t,u]*El['El_cap'] for t in m.T for u in m.U)+
                              sum(Wind['pi_omega'][o]*sum(Prices['pi_up'] [t]*m.delta_up[t,o]-Prices['pi_down'] [t]*m.delta_down[t,o] for t in m.T) for o in m.O)+
                              sum(Wind['pi_omega'][o]*sum(Prices['pi_h']*(m.h_el_u_RT[t,u,o]-m.h_el_u[t,u]) for t in m.T for u in m.U) for o in m.O)-
                              sum(Wind['pi_omega'][o]*sum(El['C_cold']*m.y_cold_RT[t,u,o]*El['El_cap'] for t in m.T for u in m.U) for o in m.O), sense=pyo.maximize)
    
    #%%  Define DA optimization constraints
    
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
        return m.p_el_u[t,u]==m.p_s[t,u] + El['P_sb']*El['El_cap']*m.z_sb[t,u]
    m.cst_p_u_tot = pyo.Constraint(m.TU, rule = cst_p_u_tot)
    
    def cst_ps_lb(m,t,u):
        return m.p_s[t,u]>=m.z_on[t,u]*El['P_min']*El['El_cap']
    m.cst_ps_lb = pyo.Constraint(m.TU, rule = cst_ps_lb)
    
    def cst_ps_ub(m,t,u):
        return m.p_s[t,u]<=m.z_on[t,u]*El['El_cap']
    m.cst_ps_ub= pyo.Constraint(m.TU, rule = cst_ps_ub)
    
    # Hydrogen production
    def cst_h_prod(m, t, u, s):
        return m.h_el_u[t,u] <= m.p_s[t,u]*El['a'][s]+m.z_on[t,u]*El['b'][s]
    m.cst_h_prod = pyo.Constraint(m.TUS, rule = cst_h_prod)
    
    #%%  Define RT optimization constraints
    
    def cst_bal_RT(m, t, o):
        return (Wind['P_w'][t]-Wind['P_w_omega'][t,o]) + (m.delta_up[t,o] - m.delta_down[t,o]) + (sum(m.p_el_u_RT[t,u,o]-m.p_el_u[t,u] for u in m.U)) == 0
    m.cst_bal_RT = pyo.Constraint(m.TO, rule=cst_bal_RT) 
    
    def cst_bal_RT_max(m, t, o):
        return  sum(m.p_el_u_RT[t,u,o] for u in m.U)<= Wind['P_w_omega'][t,o]
    m.cst_bal_RT_max = pyo.Constraint(m.TO, rule=cst_bal_RT_max)
    
    # Maximum hydrogen demand (daily or weekly)

    def cst_h_d_max_RT(m, d, o):
        return sum(m.h_el_u_RT[t,u, o] for t in TT[d] for u in m.U) <=  Demand['H_d_max_period']
    m.cst_h_d_max_RT = pyo.Constraint(m.DO, rule=cst_h_d_max_RT)
                          
   
    ############# Eletrolyzer constraints ##################
         
    def cst_bin_RT(m,t,u,o): 
        return m.z_on_RT[t,u,o] + m.z_sb_RT[t,u,o] + m.z_off_RT[t,u,o] == 1
    m.cst_bin_RT = pyo.Constraint(m.TUO, rule=cst_bin_RT)
    
    def cst_p_el_lb_u_RT(m,t,u,o): 
         return m.p_el_u_RT[t,u,o]>=El['P_min']*El['El_cap']*m.z_on_RT[t,u,o]+El['P_sb']*El['El_cap']*m.z_sb_RT[t,u,o]
    m.cst_p_el_lb_u_RT = pyo.Constraint(m.TUO, rule=cst_p_el_lb_u_RT)
    
    def cst_p_el_ub_u_RT(m,t,u,o): 
         return m.p_el_u_RT[t,u,o]<=El['P_sb']*El['El_cap']*m.z_sb_RT[t,u,o]+El['El_cap']*m.z_on_RT[t,u,o]
    m.cst_p_el_ub_u_RT= pyo.Constraint(m.TUO, rule=cst_p_el_ub_u_RT)
    
    def cst_cold_start_RT(m,t,u,o): 
        if t == m.T[1]:
            return m.y_cold_RT[t,u,o]==0
        else:
            return m.y_cold_RT[t,u,o]>=m.z_off_RT[t-1,u,o]+m.z_on_RT[t,u,o]+m.z_sb_RT[t,u,o]-1
    m.cst_cold_start_RT = pyo.Constraint(m.TUO, rule = cst_cold_start_RT)
    
    #  Power consumption
    def cst_p_u_tot_RT(m,t,u,o):
        return m.p_el_u_RT[t,u,o]==m.p_s_RT[t,u,o] + El['P_sb']*El['El_cap']*m.z_sb_RT[t,u,o]
    m.cst_p_u_tot_RT = pyo.Constraint(m.TUO, rule = cst_p_u_tot_RT)
    
    def cst_ps_lb_RT(m,t,u,o):
        return m.p_s_RT[t,u,o]>=m.z_on_RT[t,u,o]*El['P_min']*El['El_cap']
    m.cst_ps_lb_RT = pyo.Constraint(m.TUO, rule = cst_ps_lb_RT)
    
    def cst_ps_ub_RT(m,t,u,o):
        return m.p_s_RT[t,u,o]<=m.z_on_RT[t,u,o]*El['El_cap']
    m.cst_ps_ub_RT= pyo.Constraint(m.TUO, rule = cst_ps_ub_RT)
    
    # Hydrogen production
    def cst_h_prod_RT(m, t, u, s, o):
        return m.h_el_u_RT[t,u,o] <= m.p_s_RT[t,u,o]*El['a'][s]+m.z_on_RT[t,u,o]*El['b'][s]
    m.cst_h_prod_RT = pyo.Constraint(m.TUSO, rule = cst_h_prod_RT)
    

    #%% ############# Solve the problem ##################
    # Define solver
    Solver = pyo.SolverFactory(solver)
    if solver == 'gurobi':
        Solver.options['threads'] = config_dict['threads']
        
    SolverResults = Solver.solve(m, tee=True)
    SolverResults.write()

    if config_dict['print_model']==1:
        m.pprint()
    if solver == 'gurobi':
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
    
    
    # RT optimal vlaues
    p_el_RT =  np.reshape(np.array([sum(m.p_el_u_RT[t,u,o].value for u in m.U) for t in m.T for o in m.O]), (Prices['N_t'],Wind['N_omega']))
    delta_up =  np.reshape(np.array([m.delta_up[t,o].value for t in m.T for o in m.O]), (Prices['N_t'],Wind['N_omega']))
    delta_down =  np.reshape(np.array([m.delta_down[t,o].value for t in m.T for o in m.O]), (Prices['N_t'],Wind['N_omega']))
    h_el_RT =  np.reshape(np.array([sum(m.h_el_u_RT[t,u,o].value for u in m.U) for t in m.T for o in m.O]), (Prices['N_t'],Wind['N_omega']))
    
    p_el_u_RT = np.reshape(np.array([m.p_el_u_RT[t,u,o].value for t in m.T for u in m.U for o in m.O]), (Prices['N_t'],El['N_u'],Wind['N_omega']))
    h_el_u_RT = np.reshape(np.array([m.h_el_u_RT[t,u,o].value for t in m.T for u in m.U for o in m.O]), (Prices['N_t'],El['N_u'],Wind['N_omega']))
    
    p_s_RT = np.reshape(np.array([m.p_s_RT[t,u,o].value for t in m.T for u in m.U for o in m.O]), (Prices['N_t'],El['N_u'],Wind['N_omega'])) 
    
    df = pd.DataFrame(np.concatenate((c_el, P_w, p_u, p_el, h_el, delta_down, delta_up, p_el_RT, h_el_RT, z_on, z_sb, z_off, y_cold), axis=1),
                      columns =['C_el','P_w','p_u','p_el','h_el']+['delta_down']*Wind['N_omega']+['delta_up']*Wind['N_omega']+['p_el_RT']*Wind['N_omega']+['h_el_RT']*Wind['N_omega']+ ['z_on']*El['N_u'] + ['z_sb']*El['N_u'] + ['z_off']*El['N_u'] + ['y_cold']*El['N_u'])

    xi = None # variable not defined for MILP but only for MISOCP
    xi_RT = None # variable not defined for MILP but only for MISOCP
    z_s = None
    z_s_RT = None
    
        
    z_on_RT = np.reshape(np.array([m.z_on_RT[t,u,o].value for t in m.T for u in m.U for o in m.O]), (Prices['N_t'],El['N_u'],Wind['N_omega']))
    z_off_RT = np.reshape(np.array([m.z_off_RT[t,u,o].value for t in m.T for u in m.U for o in m.O]), (Prices['N_t'],El['N_u'],Wind['N_omega']))
    z_sb_RT = np.reshape(np.array([m.z_sb_RT[t,u,o].value for t in m.T for u in m.U for o in m.O]), (Prices['N_t'],El['N_u'],Wind['N_omega'])) 
    y_cold_RT = np.reshape(np.array([m.y_cold_RT[t,u,o].value for t in m.T for u in m.U for o in m.O]), (Prices['N_t'],El['N_u'],Wind['N_omega']))
        
    return obj_val, comp_time, df, p_el, p_el_u, p_u, h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi, p_el_RT, delta_up, delta_down, h_el_RT, p_el_u_RT, h_el_u_RT, z_s_RT, p_s_RT, xi_RT, z_on_RT, z_off_RT, z_sb_RT, y_cold_RT

