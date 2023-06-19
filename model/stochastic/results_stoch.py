import numpy as np
import matplotlib.pyplot as plt 

def plot_results(config_dict,El,St,Wind,Demand,Prices, obj_val, p_u, p_el, h_el, z_on, z_sb, z_off, y_cold, df, z_s, p_s, comp_time ):
    N_t = len(Prices['pi_e']) # Number of hours in the time horizon
    T = np.array([t for t in range(0, N_t)])
    N_print = 30*24 # print up tp 1 month
    #N_print = N_t
    
    fig, ax1 = plt.subplots() 
      
    ax1.set_xlabel('Time [h]') 
    ax1.set_ylabel('Electricity prices [€/MWh]', color = 'black') 
    plot_1 = ax1.plot(T[:N_print], Prices['pi_e'][:N_print], '^',color = 'black') 
    ax1.tick_params(axis ='y', labelcolor = 'black') 

    #ax1.set_xlim([0, 23])
    # Adding Twin Axes

    ax2 = ax1.twinx() 

    ax2.set_ylabel('DA electrolyzer power [MW]', color = 'green') 
    ax2.axhline(y=0, color='r', linestyle='--', label='Off')
    ax2.axhline(y=El['P_sb']*El['El_cap_tot'], color='b', linestyle='--', label='Standby')
    plot_2 = ax2.step(T[:N_print], p_el[:N_print], color = 'green', where="post")  
    plot_3 = ax2.step(T[:N_print], Wind['P_w'][:N_print], color = 'gray', where="post", label='Wind production') 
    ax2.tick_params(axis ='y', labelcolor = 'green') 
    #ax2.set_ylim([0, 1.1])
    plt.grid()
    plt.legend()
     
    # Plot status
    U_on = np.sum(z_on, axis = 1)
    U_sb = np.sum(z_sb, axis = 1)
    U_off = np.sum(z_off, axis = 1)
    
    fig_status, axxx = plt.subplots()
     
    # plot bars in stack manner
    plt.bar(T[:N_print], U_on[:N_print], color = 'g')
    plt.bar(T[:N_print], U_sb[:N_print], bottom=U_on, color='b')
    plt.bar(T[:N_print], U_off[:N_print], bottom=U_on+U_sb, color='r')
 
    plt.xlabel("Time [h]")
    plt.ylabel("Number of units")
    plt.legend(["On", "Standby", "Off"])
    #plt.title("Status of units")
    plt.show()
    