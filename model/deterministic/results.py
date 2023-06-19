import numpy as np
import matplotlib.pyplot as plt 
import matplotlib

def plot_results(config_dict,El,St,Wind,Demand,Prices, obj_val, p_u, p_el, h_el, z_on, z_sb, z_off, y_cold, df, z_s, p_s, comp_time):
    font = {'family' : 'STIXGeneral',
            'size'   : 18}
    matplotlib.rc('font', **font)
    plt.rc('axes', axisbelow=True)
    
    N_t = len(Prices['pi_e']) # Number of hours in the time horizon
    T = np.array([t for t in range(0, N_t)])
    N_print = 200 # print max 200 hours
    
    fig, ax1 = plt.subplots() 
      
    ax1.set_xlabel('Time [h]') 
    ax1.set_ylabel('Electricity prices [€/MWh]', color = 'black') 
    plot_1 = ax1.plot(T[:N_print], Prices['pi_e'][:N_print], '^',color = 'black', label='Electricity prices' ) 
    ax1.tick_params(axis ='y', labelcolor = 'black') 

    # Adding Twin Axes

    ax2 = ax1.twinx() 
    plot_3 = ax2.step(T[:N_print], Wind['P_w'][:N_print], color = 'gray', where="post", label='Wind production')  
    ax2.set_ylabel('Power [MW]', color = 'black') 
    #ax2.axhline(y=0, color='r', linestyle='--', label='Off')
    ax2.axhline(y=El['P_sb']*El['El_cap_tot'], color='b', linestyle='--')
    #ax2.axhline(y=El['P_eta_max']*El['El_cap_tot'], color='y', linestyle='--', label='Maximum efficiency')
    plot_2 = ax2.step(T[:N_print], p_el[:N_print], color = 'green', where="post", label='Electrolyzer consumption') 
    #ax2.axhline(y=0.15, color='w', linestyle='--')

    ax2.tick_params(axis ='y', labelcolor = 'black') 
    #ax2.set_ylim([0, 1.1])
    #plt.grid()
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    lines = lines_1 + lines_2 
    labels = labels_1 + labels_2 
    plt.legend(lines,labels,bbox_to_anchor=(1.15,-0.2), edgecolor='black', fancybox=False, ncol =2)
    #plt.legend()

    # Plot status
    U_on = np.sum(z_on, axis = 1)
    U_sb = np.sum(z_sb, axis = 1)
    U_off = np.sum(z_off, axis = 1)
    
    fig_status, axxx = plt.subplots()
     
    # plot bars in stack manner
    plt.bar(T[:N_print], U_on[:N_print], color = 'g')
    plt.bar(T[:N_print], U_sb[:N_print], bottom=U_on[:N_print], color='b')
    plt.bar(T[:N_print], U_off[:N_print], bottom=U_on[:N_print]+U_sb[:N_print], color='r')
 
    plt.xlabel("Time [h]")
    plt.ylabel("Number of units")
    plt.legend(["On", "Standby", "Off"])
    #plt.title("Status of units")
    plt.show()
    
    
def check_tightness(config_dict,El, Prices, p_el, p_el_u, p_s, xi, z_on, h_el_u, z_s):
        
    if config_dict["eff_type"]== 3: # HYP-SOC
        err = xi-p_s**2
        gap = np.max(np.absolute(err))
        h_physics = El['D_0']*z_on+El['D_1']*p_s+El['D_2']*p_s**2
        
    elif config_dict["eff_type"]== 2:  # HYP-L
        h_equal = np.zeros((Prices['N_t'],El['N_u'],El['N_s']))

        for s in range (0,El['N_s']):
            h_equal[:,:,s]= p_s*El['a'][s]+z_on*El['b'][s]
        h_equal_min =  np.min(h_equal, axis = 2)
        err = h_equal_min - h_el_u
        gap = np.max(np.absolute(err))
        h_physics = h_equal_min
        
    elif config_dict["eff_type"]== 4:  # HYP-MISOC
        err = xi-p_s**2
        gap = np.max(np.absolute(err))
        h_physics = El['D_0']*z_s[:,:,0]+El['D_1']*p_s[:,:,0]+El['D_2']*p_s[:,:,0]**2 + El['E_0']*z_s[:,:,1]+El['E_1']*p_s[:,:,1]+El['E_2']*p_s[:,:,1]**2
    else:
        err = None
        gap = None
        h_physics = None
        
    return err, gap, h_physics
