import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
import time

def full_pep_from_ninemers(epi_list):
    if not epi_list:
        return ""
    full_peptide = epi_list[0]
    for i in range(1, len(epi_list)):
        full_peptide += epi_list[i][-1]
    return full_peptide

def get_nth_element(input_str, n):
    elements = input_str.strip('()').split(',')
    return elements[n].strip()

def string_to_numeric_tuple(input_str):
    elements = input_str.strip('()').split(',')
    return tuple(map(float, elements))

def create_evenly_spaced_list(L, U, N):
    return np.linspace(L, U, N).tolist()

def create_log_spaced_list(L, U, N):
    return np.logspace(np.log10(L), np.log10(U), N).tolist()

def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def load_and_preprocess_data(args):
    W = load_data(os.path.join(args.tesla_variables_dir, f'TESLA_variables_{args.distance_metric_type}.pkl'))
    V = load_data(os.path.join(args.iedb_variables_dir, f'Koncz_IEDB_variables_{args.distance_metric_type}.pkl'))

    tesla_data = W[0]
    tesla_data['assay'] = 'TESLA'

    iedb_data = V[0]
    iedb_data['assay'] = 'IEDB'

    combined_data = pd.concat([tesla_data, iedb_data]).sort_values(by='peptides')
    hla_df = combined_data[combined_data['allele'] == args.allele]
    return hla_df

def create_evenly_spaced_list(L, U, N):
    return np.linspace(L, U, N).tolist()
def create_log_spaced_list(L, U, N):
    return np.logspace(np.log10(L), np.log10(U), N).tolist()

# # The parameter sets used in Rust.
gamma_d_self_values = sorted(list(set( list(np.round(create_evenly_spaced_list(1e-6, 1e-4, 8),10)) + list(np.round(create_log_spaced_list(2e-4, 1, 7),4)) )))
gamma_logkd_self_values = sorted(list(set( list(np.round(create_log_spaced_list(1e-2, 1.0, 5),4)) + list(np.round(create_log_spaced_list(5e-3, 0.6, 12),4))  )))

gamma_d_nonself_values = sorted(list(set( list(np.round(create_evenly_spaced_list(1e-6, 1e-4, 8),10)) + list(np.round(create_log_spaced_list(1e-1, 5, 6),4)) + [1e-8, 1e-100])))
gamma_logkd_nonself_values = sorted(list(set( list(np.round(create_log_spaced_list(1e-2, 1.0, 5),4)) + list(np.round(create_log_spaced_list(5e-3, 0.6, 10),4))  +[1e-8, 1e-100])))

'''

FOR CONTEXT ON WHAT THIS FILE IS MEANT TO ACCOMPLISH, SEE MY EMAIL 
TO MARTA (HER SINAI ADDRESS, MY GMAIL) FROM JUNE 20, 2024.


'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-pkl_dir", default="/Users/marcus/Work_Data/rust_outputs_local/immunogenicity_outputs/d_ub_100_d_lb_0") 
    parser.add_argument("-distance_metric_type", default="all_tcr_all_combos_model")
    parser.add_argument("-tesla_variables_dir", default="/Users/marcus/Work_Data/Minerva_editing/CFIT_Editing/bin/TESLA")
    parser.add_argument("-iedb_variables_dir", default="/Users/marcus/Work_Data/Minerva_editing/CFIT_Editing/bin/IEDB")
    parser.add_argument("-allele", default='A0201')
    args = parser.parse_args()

    start_time = time.time()
    hla_df = load_and_preprocess_data(args)
    nb_records = []
    
    imm_info_dict = {}
    for idx, (df_index,row) in enumerate(hla_df.iterrows()):
        query_epi_list = row['peptides']
        full_query_epi = full_pep_from_ninemers(query_epi_list)
        strg = f'immunogenicity_outputs_{args.distance_metric_type}_{idx}_' + '_'.join(query_epi_list) + f'_HLA-{args.allele}.pkl'
        inputs = load_data(os.path.join(args.pkl_dir, args.allele, strg))

        '''
        outputs = [logKInv_entropy_self_dict, logCh_dict, logKInv_entropy_Koncz_imm_epi_dict, logKInv_entropy_Koncz_non_imm_epi_dict, 
                   logKInv_entropy_Ours_imm_epi_dict, logKInv_entropy_Ours_non_imm_epi_dict,
                   log_rho_dict]
        '''
        logKInv_entropy_self_dict, logCh_dict, logKInv_entropy_Koncz_imm_epi_dict, logKInv_entropy_Koncz_non_imm_epi_dict, logKInv_entropy_Ours_imm_epi_dict, logKInv_entropy_Ours_non_imm_epi_dict, log_rho_dict = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]
        
        for query_epi in log_rho_dict:
            for SELF_params, foreign_dict in log_rho_dict[query_epi].items():
                SELF_params_numeric = string_to_numeric_tuple(SELF_params)
                
                if SELF_params_numeric[0] != 1 or SELF_params_numeric[1] != 1:
                    continue 
                
                if SELF_params_numeric not in imm_info_dict:
                    imm_info_dict[SELF_params_numeric] = {}
                    
                gamma_logKd_str = get_nth_element(SELF_params, 1)
                log_Ch = logCh_dict[gamma_logKd_str if gamma_logKd_str != '1.0' else '1']
                log_non_rho = log_Ch + logKInv_entropy_self_dict[query_epi][SELF_params][0] - logKInv_entropy_self_dict[query_epi][SELF_params][1]

                for FOREIGN_params, log_rho in foreign_dict.items():
                    FOREIGN_params_numeric = string_to_numeric_tuple(FOREIGN_params)
                    
                    # # First foreign paramset of interest
                    if FOREIGN_params_numeric[0] == 1e-100 and FOREIGN_params_numeric[1] == 1:
                        if FOREIGN_params_numeric not in imm_info_dict[SELF_params_numeric]:
                             imm_info_dict[SELF_params_numeric][FOREIGN_params_numeric] = {} 
                        if idx not in imm_info_dict[SELF_params_numeric][FOREIGN_params_numeric]:
                            imm_info_dict[SELF_params_numeric][FOREIGN_params_numeric][idx] = (0,0,0,0,0,0)
                        
                        nb = np.exp(log_non_rho + log_rho) 
                        KselfInv_div_Entropy = np.exp(-np.diff(logKInv_entropy_self_dict[query_epi][SELF_params])[0])
                        K_konczInv_imm_div_Entropy = np.exp(-np.diff(logKInv_entropy_Koncz_imm_epi_dict[query_epi][FOREIGN_params])[0])
                        K_konczInv_non_imm_div_Entropy = np.exp(-np.diff(logKInv_entropy_Koncz_non_imm_epi_dict[query_epi][FOREIGN_params])[0])
                        K_oursInv_imm_div_Entropy = np.exp(-np.diff(logKInv_entropy_Ours_imm_epi_dict[query_epi][FOREIGN_params])[0])
                        K_oursInv_non_imm_div_Entropy = np.exp(-np.diff(logKInv_entropy_Ours_non_imm_epi_dict[query_epi][FOREIGN_params])[0])
                        
                        # # Store the greatest nb value for this long query epi (for this .pkl)
                        if nb > imm_info_dict[SELF_params_numeric][FOREIGN_params_numeric][idx][0]:
                            imm_info_dict[SELF_params_numeric][FOREIGN_params_numeric][idx] = (nb, KselfInv_div_Entropy, K_konczInv_imm_div_Entropy, K_konczInv_non_imm_div_Entropy, K_oursInv_imm_div_Entropy, K_oursInv_non_imm_div_Entropy)
                            
                    # # Second foreign paramset of interest
                    if FOREIGN_params_numeric[0] == 1e-100 and FOREIGN_params_numeric[1] == 1e-100:
                        if FOREIGN_params_numeric not in imm_info_dict[SELF_params_numeric]:
                             imm_info_dict[SELF_params_numeric][FOREIGN_params_numeric] = {} 
                        if idx not in imm_info_dict[SELF_params_numeric][FOREIGN_params_numeric]:
                            imm_info_dict[SELF_params_numeric][FOREIGN_params_numeric][idx] = (0,0,0,0,0,0)
                        
                        nb = np.exp(log_non_rho + log_rho) 
                        KselfInv_div_Entropy = np.exp(-np.diff(logKInv_entropy_self_dict[query_epi][SELF_params])[0])
                        K_konczInv_imm_div_Entropy = np.exp(-np.diff(logKInv_entropy_Koncz_imm_epi_dict[query_epi][FOREIGN_params])[0])
                        K_konczInv_non_imm_div_Entropy = np.exp(-np.diff(logKInv_entropy_Koncz_non_imm_epi_dict[query_epi][FOREIGN_params])[0])
                        K_oursInv_imm_div_Entropy = np.exp(-np.diff(logKInv_entropy_Ours_imm_epi_dict[query_epi][FOREIGN_params])[0])
                        K_oursInv_non_imm_div_Entropy = np.exp(-np.diff(logKInv_entropy_Ours_non_imm_epi_dict[query_epi][FOREIGN_params])[0])
                        
                        # # Store the greatest nb value for this long query epi (for this .pkl)
                        if nb > imm_info_dict[SELF_params_numeric][FOREIGN_params_numeric][idx][0]:
                            imm_info_dict[SELF_params_numeric][FOREIGN_params_numeric][idx] = (nb, KselfInv_div_Entropy, K_konczInv_imm_div_Entropy, K_konczInv_non_imm_div_Entropy, K_oursInv_imm_div_Entropy, K_oursInv_non_imm_div_Entropy)
    
    # # REFACTOR THE DICT VALUES INTO EASILY VISUALISABLE DATAFRAME TABLE
    for SELF_params, foreign_dict in log_rho_dict[query_epi].items():
        SELF_params_numeric = string_to_numeric_tuple(SELF_params)
        
        if SELF_params_numeric[0] != 1 or SELF_params_numeric[1] != 1:
            continue 
            
        
        for FOREIGN_params, _ in foreign_dict.items():
            FOREIGN_params_numeric = string_to_numeric_tuple(FOREIGN_params)
            
            # # First foreign paramset of interest
            if FOREIGN_params_numeric[0] == 1e-100 and FOREIGN_params_numeric[1] == 1:
                nb_records = list(imm_info_dict[SELF_params_numeric][FOREIGN_params_numeric].values())
                nb_df = pd.DataFrame(nb_records, columns=['nb', 'KselfInv_div_Entropy', 'K_konczInv_imm_div_Entropy', 'K_konczInv_non_imm_div_Entropy', 'K_oursInv_imm_div_Entropy', 'K_oursInv_non_imm_div_Entropy'])
                imm_info_dict[SELF_params_numeric][FOREIGN_params_numeric] = nb_df
                
            # # Second foreign paramset of interest
            if FOREIGN_params_numeric[0] == 1e-100 and FOREIGN_params_numeric[1] == 1e-100:
                nb_records = list(imm_info_dict[SELF_params_numeric][FOREIGN_params_numeric].values())
                nb_df = pd.DataFrame(nb_records, columns=['nb', 'KselfInv_div_Entropy', 'K_konczInv_imm_div_Entropy', 'K_konczInv_non_imm_div_Entropy', 'K_oursInv_imm_div_Entropy', 'K_oursInv_non_imm_div_Entropy'])
                imm_info_dict[SELF_params_numeric][FOREIGN_params_numeric] = nb_df
                

                    
