import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt 
from scipy.optimize import minimize

################# Constant parameters ################# 
F = 96485.3 # Faraday constant [sA/mol]
M_h2 = 2.0159e-3 # Molar mass of hydrogen [kg/mol]

################# Electrolyzer parameters ################# 
T_op = 90 # Operating temperature  [C]
Pr = 30 # Operating pressure  [bar]
i_max = 5000 # Maximum current density [A/m2]

# Coefficients from Sanchez paper
r1 = 4.45153e-5 # Electrolyte ohmic resistive parameter r1 [Ohm m2]
r2 = 6.88874e-9 # Electrolyte ohmic resistive parameter r2 [Ohm m2 C-1]
d1 = -3.12996e-6 # [Ohm m2]
d2 = 4.47137e-7 # [Ohm m2 bar-1]
s = 0.33824 # Over voltage parameter of electrode [V] 
t1 = -0.01539 # Over voltage parameter of electrode [A-1 m2]  
t2 = 2.00181 # Over voltage parameter of electrode [A-1 m2 C]
t3 = 15.24178 # Over voltage parameter of electrode [A-1 m2 C^2]

# Faraday efficiency - Formulataion with 4 parameters (based on Ulleberg)
f11 = 478645.74 # A2 m-4
f12 = -2953.15 # A2 m-4 C-1
f21 = 1.03960
f22 = -0.00104 #C-1 
 
################# Electrolyzer functions ################# 

# Open circuit voltage as a function of temperature (at standard pressure) 
def U_rev(T):
    U_rev = 1.5184-1.5421*1e-3*(T+273.15)+9.523*1e-5*(T+273.15)*np.log(T+273.15)+9.84*1e-8*(T+273.15)**2 
    return(U_rev)
# Voltage as a function of current density, temperature and pressure
def U_cell(i,T,p):
    U_cell = U_rev(T)+(r1+d1+r2*T+d2*p)*i+s*np.log10((t1+t2/T+t3/T**2)*i+1) # Temperature in Celsium
    return(U_cell)
# Cell power density consumption as a function of temperature and current density [W/m2]
def P_cell(i, T, p):
    P_cell = U_cell(i, T, p) * i
    return(P_cell)
# Calculate the total cell area (necessary to have the desired electrolyzer capacity)
def area(El):
    AA = El["El_cap"]*10**(6)/P_cell(i_max, T_op, Pr)  #m2
    return AA
# Power consumption of the stack [MW]
def P_stack(i, T, p, AA):
    P_stack = P_cell(i, T, p) * AA * 10**(-6)
    return(P_stack)
# Faraday efficiency
def eta_farad(i,T):
    eta_farad = (i**2./(f11+f12*T+i**2))*(f21+f22*T)
    return(eta_farad)
# Hydrogen production kg/h
def h_prod(i,T,p,AA):
    h_prod = eta_farad(i,T)*M_h2*i/(2*F) * AA * 3600
    return h_prod
# Function for efficiency as a function of current desnity, temperature, pressure
def eta(i,T,p,AA):
    eta = h_prod(i,T,p,AA)/P_stack(i,T,p,AA)
    return eta

# Function to find the current corresponding to a certain power value [MW]
def find_i_from_p(p_val,AA,T):
    i_pval = np.zeros(np.size(p_val))
    for j in range(np.size(p_val)):
        cc=p_val[j]
        def P_stack_cc(i, T=T_op, p=Pr):
            P_stack_cc = P_cell(i, T, p) * AA * 10**(-6) - cc
            return(P_stack_cc)
        i_pval[j] = fsolve(P_stack_cc,[2000])
    return i_pval

# Function to find the power corresonding to the maximum efficiency
def p_eta_max_fun(El):
    AA = area(El)
    i_min = find_i_from_p(np.array([El["P_min"]*El["El_cap"]]),AA,T_op)
    i_min_max = np.linspace(i_min, i_max, num=500)
    eta_max_pos=np.argmax(eta(i_min_max, T_op, Pr, AA))
    i_eta_max=i_min_max[eta_max_pos]
    P_eta_max=P_stack(i_eta_max,T_op, Pr,AA)/El["El_cap"] # 0.28 at 90 degrees! But it changes with T!
    eta_max=eta(i_eta_max,T_op, Pr, AA) # 19.78 at 90 degrees
    El.update({'P_eta_max': P_eta_max[0] }), El.update({'eta_max': eta_max[0] })
    return P_eta_max[0]

# Function to find the linear coefficients 
def lin_coeff(x,y):
    coeff = np.polyfit(x,y, 1)
    # y=coeff[1]+coeff[0]*x
    return coeff[1], coeff[0]

# Function to find the quadratic coefficients 
def lin_coeff_2(x,y):
    coeff = np.polyfit(x,y, 2)
    # y=coeff[1]+coeff[0]*x
    return coeff

def objective(Q, x_data, y_data, El):
    # Extract polynomial coefficients
    Q_2, Q_1, Q_0 = Q
    
    # Evaluate polynomial
    y_model = Q_2*x_data**2 + Q_1*x_data + Q_0
    
    # Calculate sum of squared errors
    residuals = y_data - y_model
    sse = np.sum(residuals**2)
    
    # Add residual for violation of having the same p_eta_max
    sse += 10000*(Q_0 - x_data[np.abs(x_data - p_eta_max_fun(El)*El['El_cap']).argmin()]**2 * Q_2)**2

    return sse

def objective_rhs(Q, x_data, y_data, p_eff_target, El):
    # Extract polynomial coefficients
    Q_2, Q_1, Q_0 = Q
    # Evaluate polynomial
    y_model = Q_2*x_data**2 + Q_1*x_data + Q_0
    # Calculate sum of squared errors
    residuals = y_data - y_model
    sse = np.sum(residuals**2)
    
    AA = area(El)
    i_target = find_i_from_p(np.array([p_eff_target]),AA,T_op)
    p_eff_target_pos = np.abs(x_data - p_eff_target).argmin()
    sse += 100000*(Q_2*x_data[p_eff_target_pos] + Q_1 + Q_0/x_data[p_eff_target_pos] - eta(i_target,T_op,Pr,AA))**2
    sse += 100000*(Q_2*x_data[-1] + Q_1 + Q_0/x_data[-1] - El['eta_full_load'])**2

    return sse

#####################################################################################################

def initialize_electrolyzer(El, config_dict):
    AA = area(El)
    El.update({'Area': AA })
    El.update({'eta_full_load': eta(i_max,T_op,Pr,AA)})

    if config_dict["eff_type"]== 1: # HYP-MIL
        i_val = find_i_from_p(El["p_val"],AA,T_op) # Calculate the corresponding current densities
        N_p=len(El["p_val"]) # Number of piecewise discretization points
        N_s = N_p - 1 # Number of discretization segments
        El.update({'N_p': N_p }), El.update({'N_s': N_s })
        
        # Calculate coefficients
        a=np.zeros(N_s)
        b=np.zeros(N_s)
        
        for jj in range(0, N_s):
            x=np.concatenate((np.array([El["p_val"][jj]]),np.array([El["p_val"][jj+1]])))
            y=np.concatenate((np.array([h_prod(i_val[jj],T_op,Pr,AA)]),np.array([h_prod(i_val[jj+1],T_op,Pr,AA)])))
            b[jj], a[jj] = lin_coeff(x,y)
        Q_0 = 0
        Q_1 = 0
        
    elif config_dict["eff_type"]== 2:  # HYP-L
        i_val = find_i_from_p(El["p_val"],AA,T_op) # Calculate the corresponding current densities
        N_p=len(El["p_val"]) # Number of piecewise discretization points
        N_s = N_p - 1 # Number of discretization segments
        El.update({'N_p': N_p }), El.update({'N_s': N_s })
        
        # Calculate coefficients
        a=np.zeros(N_s)
        b=np.zeros(N_s)
        
        for jj in range(0, N_s):
            x=np.concatenate((np.array([El["p_val"][jj]]),np.array([El["p_val"][jj+1]])))
            y=np.concatenate((np.array([h_prod(i_val[jj],T_op,Pr,AA)]),np.array([h_prod(i_val[jj+1],T_op,Pr,AA)])))
            b[jj], a[jj] = lin_coeff(x,y)
        Q_0 = 0
        Q_1 = 0
                
    elif config_dict["eff_type"]== 3: # HYP-SOC
        El.update({'N_p': 2 }),El.update({'N_s': 1 })
        a = 0
        b = 0
        Q_0 = 0
        Q_1 = 0
        
        AA = area(El)
        i_min = find_i_from_p(np.array([El["P_min"]*El["El_cap"]]),AA,T_op)
        i_min_max = np.linspace(i_min, i_max, num=500)
        p_min_max = P_stack(i_min_max , T_op, Pr, AA)
        
        quad_fit = 1 # 0: quadratic fit, 1: maximum efficiency in p_eta_max
        
        if quad_fit == 0: # Quadratic fit
            x=p_min_max.flatten()
            y=h_prod(i_min_max,T_op,Pr, AA)
            D_2,D_1,D_0 = lin_coeff_2(x,y)

        else: # Quadratic fit with maximum at P_eta_max
            res = minimize(objective, x0 = [0,0,0], args=(p_min_max.flatten(), h_prod(i_min_max, T_op, Pr, AA).flatten(),El))
            D_2, D_1, D_0 = res.x
        
        El.update({'D_2': D_2}), El.update({'D_1': D_1}), El.update({'D_0': D_0})
        

    elif config_dict["eff_type"]== 4: # HYP-MISOC
        N_p=len(El["p_val"]) # Number of piecewise discretization points
        N_s = N_p - 1 # Number of discretization segments
        El.update({'N_p': N_p }), El.update({'N_s': N_s})
        a = 0
        b = 0
        Q_0 = 0
        Q_1 = 0
        
        AA = area(El)
        i_min = find_i_from_p(np.array([El["p_val"][0]]),AA,T_op)
        i_mid = find_i_from_p(np.array([El["p_val"][1]]),AA,T_op)
        
        i_vec1 = np.linspace(i_min, i_mid, num=500)
        p_vec1 = P_stack(i_vec1 , T_op, Pr, AA)
        
        i_vec2 = np.linspace(i_mid, i_max, num=500)
        p_vec2 = P_stack(i_vec2 , T_op, Pr, AA)
        
        # Left coefficiencys
        res = minimize(objective, x0 = [0,0,0], args=(p_vec1.flatten(), h_prod(i_vec1, T_op, Pr, AA).flatten(), El))
        D_2, D_1, D_0 = res.x
        El.update({'D_2': D_2}), El.update({'D_1': D_1}), El.update({'D_0': D_0})
        # Right coefficients
        res = minimize(objective_rhs, x0 = [0,0,0], args=(p_vec2.flatten(), h_prod(i_vec2, T_op, Pr, AA).flatten(), El['p_val'][1], El))
        E_2, E_1, E_0 = res.x
        El.update({'E_2': E_2}), El.update({'E_1': E_1}), El.update({'E_0': E_0})
    
    El.update({'a': a}), El.update({'b': b}), El.update({'Q_0': Q_0}), El.update({'Q_1': Q_1})

#####################################################################################################        

def plot_el_curves(El, config_dict):
    # Colors for plotting
    black = 'black'
    red = '#d3494e'
    green = '#087804'
    blue = '#0343df'
    
    AA = area(El)
    
    # Plot the original nolinear curves
    # Find the current corresponding to the minimum allowed operating power (P_min)
    i_min = find_i_from_p(np.array([El["P_min"]*El["El_cap"]]),AA,T_op)
    i_min_max = np.linspace(i_min, i_max, num=500)
    p_min_max = P_stack(i_min_max, T_op, Pr, AA)
    
    
    # Nonlinar production curve
    fig_prod=plt.figure()
    #fig_prod=plt.gcf().number
    plt.title('Hydrogen production')
    plt.xlabel('Power [MW]')
    plt.ylabel('Hydrogen [kg/h]')
    plt.plot(p_min_max, h_prod(i_min_max, T_op, Pr, AA), label='Nonlinear', color = black)
    plt.legend()
    plt.grid()
    plt.xlim([El["P_min"]*El["El_cap"], El["El_cap"]])
    
    # Nonlinar eficiency
    fig_eff=plt.figure()
    #fig_eff=plt.gcf().number
    plt.title('Efficiency curve' )
    plt.xlabel('Power [MW]')
    plt.ylabel('Efficiency [kg/MWh]')
    plt.plot(p_min_max, eta(i_min_max, T_op, Pr, AA),label='Nonlinear', color = black)
    plt.legend()
    plt.grid()
    plt.xlim([El["P_min"]*El["El_cap"], El["El_cap"]])
    
    if config_dict["eff_type"]== 1:   #HYP-MIL
        N_pw=len(El["p_val"]) # Number of piecewise discretization points
        N_s = N_pw - 1 # Number of discretization segments

        # Approximated hydorgen production
        for jj in range(0, N_s):
            x=np.concatenate((np.array([El["p_val"][jj]]),np.array([El["p_val"][jj+1]])))
            # Plot hydrogen production in each segment
            plt.figure(fig_prod)
            plt.plot(x, (El["b"][jj]+El["a"][jj]*x), ":.", color='red', linewidth=2, label='Piecewise linearization', markersize=10)  
        plt.legend(['Nonlinear curve', 'Approximated curve'])
        # Approximated efficiency curve
        for jj in range(0, N_s):
            p_vec=np.linspace(El["p_val"][jj],El["p_val"][jj+1])
            plt.figure(fig_eff)
            plt.plot(p_vec, (El["a"][jj]+El["b"][jj]/(p_vec)), color='red', linewidth=2, label='Piecewise linearization', linestyle='--', dashes=(1, 1)) 
        plt.legend(['Nonlinear curve', 'Approximated curve'])
        
    if config_dict["eff_type"]==2: #HYP_L  
        N_pw=len(El["p_val"]) # Number of piecewise discretization points
        N_s = N_pw - 1 # Number of discretization segments

        # Approximated hydorgen production
        for jj in range(0, N_s):
            x=np.concatenate((np.array([El["P_min"]*El["El_cap"]]),np.array([El["El_cap"]])))
            # Plot hydrogen production in each segment
            plt.figure(fig_prod)
            plt.plot(x, (El["b"][jj]+El["a"][jj]*x), color='red', linewidth=1, label='Approximated curve', markersize=10)  
            plt.scatter(El['p_val'][jj], (El["b"][jj]+El["a"][jj]*El['p_val'][jj]), c= 'black') 

        plt.legend(['Nonlinear curve', 'Approximated curve'])
        # Approximated efficiency curve
        for jj in range(0, N_s):
            p_vec=np.linspace(El["p_val"][jj],El["p_val"][jj+1])
            plt.figure(fig_eff)
            plt.plot(p_vec, (El["a"][jj]+El["b"][jj]/(p_vec)), color='red', linewidth=1, label='Approximated curve', linestyle='--', dashes=(1, 1)) 
        plt.legend(['Nonlinear curve', 'Approximated curve'])
    
    if config_dict["eff_type"]== 3: #HYP-SOC
            
        p_val = np.concatenate((El["p_val"][:1], [El["p_val"][-1]])) # new p_val vector with only relevant points 
        N_pw=len(p_val) # Number of discretization points
        N_s = N_pw - 1 # Number of regions

        # Approximated hydorgen production
        for jj in range(0, N_s):
            p_vec=np.linspace(p_val[jj],p_val[jj+1])
            plt.figure(fig_prod)
            plt.plot(p_vec, (El["D_0"]+El["D_1"]*p_vec+El["D_2"]*p_vec**2),":", color='blue', linewidth=2, label='Quadratic', markersize=10)    
        plt.legend()
        # Approximated efficiency curve
        for jj in range(0, N_s):

            p_vec=np.linspace(p_val[jj],p_val[jj+1])
            plt.figure(fig_eff)
            plt.plot(p_vec, (El["D_0"]/p_vec+El["D_1"]+El["D_2"]*p_vec),":", color='blue', linewidth=2, label='Quadratic', markersize=10)
        plt.legend()
        plt.show()
        
    if config_dict["eff_type"]== 4: # HYP-MISOC
   
        p_val = El['p_val'] # new p_val vector with only relevant points 
        N_pw=len(p_val) # Number of discretization points
        N_s = N_pw - 1 # Number of regions

        # Approximated hydorgen production
        for jj in range(0, N_s):
            p_vec=np.linspace(p_val[jj],p_val[jj+1])
            plt.figure(fig_prod)
            if jj == 0:
                plt.plot(p_vec, (El["D_0"]+El["D_1"]*p_vec+El["D_2"]*p_vec**2),":", color='red', linewidth=2, label='Left', markersize=10)  
            else:
                plt.plot(p_vec, (El["E_0"]+El["E_1"]*p_vec+El["E_2"]*p_vec**2),":", color='blue', linewidth=2, label='Right', markersize=10)      
        plt.legend()
        # Approximated efficiency curve
        for jj in range(0, N_s):
            p_vec=np.linspace(p_val[jj],p_val[jj+1])
            plt.figure(fig_eff)
            if jj == 0:
                plt.plot(p_vec, (El["D_0"]/p_vec+El["D_1"]+El["D_2"]*p_vec),":", color='red', linewidth=2, label='Left', markersize=10)
            else:
                plt.plot(p_vec, (El["E_0"]/p_vec+El["E_1"]+El["E_2"]*p_vec),":", color='blue', linewidth=2, label='Right', markersize=10)
        plt.legend()
        plt.show()
            

################ Ex post ##################
def expost(config_dict,El,St,Wind,Demand,Prices, obj_val, p_el, p_el_u, h_el, z_on):

    p_el_u_prod=np.multiply(p_el_u,z_on)
    h_el_ex_post_u = np.zeros((Prices['N_t'],El['N_u']))
    for u in range(0,El['N_u']):
        i_el_u = find_i_from_p(np.array(p_el_u_prod[:,u]),El['Area'],T_op) 
        h_el_ex_post_u[:,u] = h_prod(i_el_u,T_op,Pr,El['Area']).T

    h_el_ex_post= np.sum(h_el_ex_post_u, axis=1,keepdims=True)
    
    obj_val_ex_post = obj_val + (np.sum(h_el_ex_post)-np.sum(h_el))*Prices['pi_h']
    
    obj_val_diff = (obj_val_ex_post-obj_val)/obj_val_ex_post*100
    h_el_diff =  (np.sum(h_el_ex_post)-np.sum(h_el))/np.sum(h_el)*100
    
    return obj_val_ex_post, h_el_ex_post, obj_val_diff, h_el_diff



    
    
    
