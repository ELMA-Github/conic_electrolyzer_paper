import model_op_HYP_MIL_stoch, model_op_HYP_L_stoch, model_op_HYP_SOC_stoch, model_op_HYP_MISOC_stoch

def solve_fun(config_dict,El,St,Wind,Demand,Prices):

    if config_dict['eff_type']==1: # HYP-MIL
        obj_val, comp_time, df, p_el, p_el_u, p_u, h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi,  p_el_RT, delta_up, delta_down, h_el_RT, p_el_u_RT, h_el_u_RT, z_s_RT, p_s_RT, xi_RT, z_on_RT, z_off_RT, z_sb_RT, y_cold_RT = model_op_HYP_MIL_stoch.model_HYP_MIL(config_dict,El,St,Wind,Demand,Prices)
        return obj_val, comp_time, df, p_el, p_el_u, p_u, h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi, p_el_RT, delta_up, delta_down,  h_el_RT, p_el_u_RT, h_el_u_RT, z_s_RT, p_s_RT, xi_RT, z_on_RT, z_off_RT, z_sb_RT, y_cold_RT
    
    if config_dict['eff_type']==2: # HYP-L
        obj_val, comp_time, df, p_el, p_el_u, p_u,  h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi,  p_el_RT, delta_up, delta_down,  h_el_RT, p_el_u_RT, h_el_u_RT, z_s_RT, p_s_RT, xi_RT, z_on_RT, z_off_RT, z_sb_RT, y_cold_RT = model_op_HYP_L_stoch.model_HYP_L(config_dict,El,St,Wind,Demand,Prices)
        return obj_val, comp_time, df, p_el, p_el_u, p_u, h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi, p_el_RT, delta_up, delta_down,  h_el_RT, p_el_u_RT, h_el_u_RT, z_s_RT, p_s_RT, xi_RT, z_on_RT, z_off_RT, z_sb_RT, y_cold_RT
      
    if config_dict['eff_type']==3:  # HYP-SOC
        obj_val, comp_time, df, p_el, p_el_u, p_u, h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi,  p_el_RT, delta_up, delta_down,  h_el_RT, p_el_u_RT, h_el_u_RT, z_s_RT, p_s_RT, xi_RT, z_on_RT, z_off_RT, z_sb_RT, y_cold_RT = model_op_HYP_SOC_stoch.model_HYP_SOC(config_dict,El,St,Wind,Demand,Prices)
        return obj_val, comp_time, df, p_el, p_el_u, p_u, h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi, p_el_RT, delta_up, delta_down,  h_el_RT, p_el_u_RT, h_el_u_RT, z_s_RT, p_s_RT, xi_RT, z_on_RT, z_off_RT, z_sb_RT, y_cold_RT
    
    if config_dict['eff_type']==4:  #HYP-MISOC
        obj_val, comp_time, df, p_el, p_el_u, p_u, h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi,  p_el_RT, delta_up, delta_down,  h_el_RT, p_el_u_RT, h_el_u_RT, z_s_RT, p_s_RT, xi_RT, z_on_RT, z_off_RT, z_sb_RT, y_cold_RT = model_op_HYP_MISOC_stoch.model_HYP_MISOC(config_dict,El,St,Wind,Demand,Prices)
        return obj_val, comp_time, df, p_el, p_el_u, p_u, h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi, p_el_RT, delta_up, delta_down,  h_el_RT, p_el_u_RT, h_el_u_RT, z_s_RT, p_s_RT, xi_RT, z_on_RT, z_off_RT, z_sb_RT, y_cold_RT

