# Read wind scenarios

import pandas as pd

df_CP_st = pd.read_csv('data/wp-scen-zone1.dat', delim_whitespace=True, header=None)
i = 0
# Select 24 rconsecutive rows of the dataframe
df_CP_st_24 = df_CP_st.iloc[i:i+24]

# Select number of scenarios
def data_CP_scenarios(N_t,N_omega):
    arr = df_CP_st_24.iloc[: , :N_omega].to_numpy()
    arr = arr[:N_t]
    return arr


# Determine DA forecast

# Calculate the average capacity fator over the selected scenarios
def data_CP_w(N_t,N_omega):
    df_CP_st_24_N_omega = df_CP_st.iloc[i:i+24 , :N_omega]
    CP_ave_24_N_omega = df_CP_st_24_N_omega.mean(axis=1)
    arr = CP_ave_24_N_omega.to_numpy().flatten() 
    arr = arr[:N_t]
    return arr