import model_op_HYP_MIL, model_op_HYP_L, model_op_HYP_SOC, model_op_HYP_MISOC

def solve_fun(config_dict,El,St,Wind,Demand,Prices):
     
    if config_dict['eff_type']==1: # HYP-MIL
        obj_val, comp_time, df, p_el, p_el_u, p_u,  h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi= model_op_HYP_MIL.model_HYP_MIL(config_dict,El,St,Wind,Demand,Prices)
        return obj_val, comp_time, df, p_el, p_el_u, p_u,  h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi
    
    if config_dict['eff_type']==2: # HYP-L
        obj_val, comp_time, df, p_el, p_el_u, p_u,  h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi = model_op_HYP_L.model_HYP_L(config_dict,El,St,Wind,Demand,Prices)
        return obj_val, comp_time, df, p_el, p_el_u, p_u,  h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi
    
    if config_dict['eff_type']==3: # HYP-SOC
        obj_val, comp_time, df, p_el, p_el_u, p_u,  h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi = model_op_HYP_SOC.model_HYP_SOC(config_dict,El,St,Wind,Demand,Prices)
        return obj_val, comp_time, df, p_el, p_el_u, p_u,  h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi
    
    if config_dict['eff_type']==4: #HYP-MISOC
        obj_val, comp_time, df, p_el, p_el_u, p_u,  h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi = model_op_HYP_MISOC.model_HYP_MISOC(config_dict,El,St,Wind,Demand,Prices)
        return obj_val, comp_time, df, p_el, p_el_u, p_u,  h_el, h_el_u, z_on, z_sb, z_off, y_cold, z_s, p_s, xi
    
