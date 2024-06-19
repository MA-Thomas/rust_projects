import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
import time
import ast

import immunogenicity_rust 

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

def calculate_auc(pos, neg):
    y = np.array([2] * len(pos) + [1] * len(neg))
    pred = np.concatenate([pos, neg])
    fpr, tpr, _ = metrics.roc_curve(y, pred, pos_label=2)
    return round(metrics.auc(fpr, tpr), 12)
#%%

# # The parameter sets used in Rust.
gamma_d_self_values = sorted(list(set( list(np.round(create_evenly_spaced_list(1e-6, 1e-4, 8),10)) + list(np.round(create_log_spaced_list(2e-4, 1, 7),4)) )))
gamma_logkd_self_values = sorted(list(set( list(np.round(create_log_spaced_list(1e-2, 1.0, 5),4)) + list(np.round(create_log_spaced_list(5e-3, 0.6, 12),4))  )))

gamma_d_nonself_values = sorted(list(set( list(np.round(create_evenly_spaced_list(1e-6, 1e-4, 8),10)) + list(np.round(create_log_spaced_list(1e-1, 5, 6),4)) + [1e-8, 1e-100])))
gamma_logkd_nonself_values = sorted(list(set( list(np.round(create_log_spaced_list(1e-2, 1.0, 5),4)) + list(np.round(create_log_spaced_list(5e-3, 0.6, 10),4))  +[1e-8, 1e-100])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-pkl_dir", default="/Users/marcus/Work_Data/rust_outputs_local/immunogenicity_outputs/d_ub_70.0_d_lb_3.14")
    parser.add_argument("-distance_metric_type", default="all_tcr_all_combos_model")
    parser.add_argument("-tesla_variables_dir", default="/Users/marcus/Work_Data/Minerva_editing/CFIT_Editing/bin/TESLA")
    parser.add_argument("-iedb_variables_dir", default="/Users/marcus/Work_Data/Minerva_editing/CFIT_Editing/bin/IEDB")
    parser.add_argument("-allele", default='A0301')
    args = parser.parse_args()

    start_time = time.time()
    hla_df = load_and_preprocess_data(args)

    #%% GATHER PATHS TO PICKLE FILES STORING IMMUNOGENICITY INFO
    tesla_pos_pkl_list = []
    tesla_neg_pkl_list = []
    koncz_pos_pkl_list = []
    koncz_neg_pkl_list = []
    
    for idx, (df_index,row) in enumerate(hla_df.iterrows()):
        if row['assay'] == 'TESLA' and row['immunogenicity']==1:
            query_epi_list = row['peptides']
            full_query_epi = full_pep_from_ninemers(query_epi_list)
            strg = f'immunogenicity_outputs_{args.distance_metric_type}_{idx}_' + '_'.join(query_epi_list) + f'_HLA-{args.allele}.pkl'
            tesla_pos_pkl_list.append( os.path.join(args.pkl_dir, args.allele, strg))

        elif row['assay'] == 'TESLA' and row['immunogenicity']==0:
            query_epi_list = row['peptides']
            full_query_epi = full_pep_from_ninemers(query_epi_list)
            strg = f'immunogenicity_outputs_{args.distance_metric_type}_{idx}_' + '_'.join(query_epi_list) + f'_HLA-{args.allele}.pkl'
            tesla_neg_pkl_list.append( os.path.join(args.pkl_dir, args.allele, strg))

        elif row['assay'] == 'IEDB' and row['immunogenicity']==1:
            query_epi_list = row['peptides']
            full_query_epi = full_pep_from_ninemers(query_epi_list)
            strg = f'immunogenicity_outputs_{args.distance_metric_type}_{idx}_' + '_'.join(query_epi_list) + f'_HLA-{args.allele}.pkl'
            koncz_pos_pkl_list.append( os.path.join(args.pkl_dir, args.allele, strg))

        elif row['assay'] == 'IEDB' and row['immunogenicity']==0:
            query_epi_list = row['peptides']
            full_query_epi = full_pep_from_ninemers(query_epi_list)
            strg = f'immunogenicity_outputs_{args.distance_metric_type}_{idx}_' + '_'.join(query_epi_list) + f'_HLA-{args.allele}.pkl'
            koncz_neg_pkl_list.append( os.path.join(args.pkl_dir, args.allele, strg))

###############################################################################
##%% COMPUTE IMM_DICT (rust) AND AUC_DICT (python) -> ~40 seconds

    # rst = time.time()
    # auc_dict = {}
    # tesla_imm_dicts = immunogenicity_rust.get_immunogenicity_dicts_py(tesla_pos_pkl_list,tesla_neg_pkl_list)
    # tesla_pos_imm_dict = tesla_imm_dicts[0]
    # tesla_neg_imm_dict = tesla_imm_dicts[1]
    
    # koncz_imm_dicts = immunogenicity_rust.get_immunogenicity_dicts_py(koncz_pos_pkl_list,koncz_neg_pkl_list)
    # koncz_pos_imm_dict = koncz_imm_dicts[0]
    # koncz_neg_imm_dict = koncz_imm_dicts[1]
    # print("Time to generate/load immunogenicity_dicts from rust: ", time.time()-rst)
    # # assert(1==2)
    
    # use_python_auc_function = True
    # use_rust_auc_function = True
    
    # st = time.time()
    # for foreign_key in tesla_pos_imm_dict:
    #     FOREIGN_params_numeric = string_to_numeric_tuple(foreign_key)

    #     for self_key in tesla_pos_imm_dict[foreign_key]:
    #         SELF_params_numeric = string_to_numeric_tuple(self_key)
            
            
    #         if use_python_auc_function:
    #             tesla_auc = calculate_auc(tesla_pos_imm_dict[foreign_key][self_key], tesla_neg_imm_dict[foreign_key][self_key])
    #             koncz_auc = calculate_auc(koncz_pos_imm_dict[foreign_key][self_key], koncz_neg_imm_dict[foreign_key][self_key])
    #         if use_rust_auc_function:    
    #             tesla_auc_rust = immunogenicity_rust.calculate_auc_py(tesla_pos_imm_dict[foreign_key][self_key], tesla_neg_imm_dict[foreign_key][self_key])
    #             koncz_auc_rust = immunogenicity_rust.calculate_auc_py(koncz_pos_imm_dict[foreign_key][self_key], koncz_neg_imm_dict[foreign_key][self_key])
    #             assert(tesla_auc == round(tesla_auc_rust,12))
    #             assert(koncz_auc == round(koncz_auc_rust,12))
            
    #         if FOREIGN_params_numeric[1] not in auc_dict:
    #             auc_dict[FOREIGN_params_numeric[1]] = {}
    #         if SELF_params_numeric not in auc_dict[FOREIGN_params_numeric[1]]:
    #             auc_dict[FOREIGN_params_numeric[1]][SELF_params_numeric] = []
                
    #         auc_dict[FOREIGN_params_numeric[1]][SELF_params_numeric].append([FOREIGN_params_numeric[0], tesla_auc, koncz_auc])
   
    # print("runtime to eval auc_dict in python: ", time.time()-st)
    # assert(1==4)


###############################################################################
#%% COMPUTE AUC_DICT DIRECTLY IN RUST
    num_self_params_per_iter = 30
    tesla_auc_dict = immunogenicity_rust.calculate_auc_dict_from_pickle_files_py(tesla_pos_pkl_list,tesla_neg_pkl_list, num_self_params_per_iter)
    koncz_auc_dict = immunogenicity_rust.calculate_auc_dict_from_pickle_files_py(koncz_pos_pkl_list,koncz_neg_pkl_list, num_self_params_per_iter)


    # print("tesla_auc_dict: ",tesla_auc_dict)
    print("len(tesla_auc_dict): ", len(tesla_auc_dict))
    print("len(koncz_auc_dict): ", len(koncz_auc_dict))

#%% COMBINE AUC_DICTS AND UPDATE STRUCTURE
    foreign_key_1_set = set()
    foreign_key_0_set = set()
    auc_dict = {}
    for self_key_str in tesla_auc_dict:
        self_key = ast.literal_eval(self_key_str)
        
        for foreign_key_str in tesla_auc_dict[self_key_str]:
            foreign_key = ast.literal_eval(foreign_key_str)
            
            
            if foreign_key[1] not in auc_dict:
                print("foreign_key[1]: ",foreign_key[1])
                auc_dict[foreign_key[1]] = {}
                
            if self_key not in auc_dict[foreign_key[1]]: 
                auc_dict[foreign_key[1]][self_key] = []

            
            tesla_auc = tesla_auc_dict[self_key_str][foreign_key_str]
            koncz_auc = koncz_auc_dict[self_key_str][foreign_key_str]
            
            auc_dict[foreign_key[1]][self_key].append( (foreign_key[0],tesla_auc, koncz_auc) )
            

#%%
    keys_ranking_list = []
    for foreign_gamma_logKd in auc_dict:
        for self_params_key in auc_dict[foreign_gamma_logKd]:
            # Sort the values (lists) within each key based on the fourth element (koncz_auc)
            auc_dict[foreign_gamma_logKd][self_params_key].sort(key=lambda x: x[1], reverse=True)

            # #vConvert the list to a DataFrame
            df = pd.DataFrame(auc_dict[foreign_gamma_logKd][self_params_key], columns=['gamma_d_nonself', 'tesla_auc', 'koncz_auc'])
            
            
            '''
            Provide a ranking for [key=gamma_logKd_nonself, self_key=SELF_params].
            A higher ranking implies that:  
            a particular gamma_d_nonself value does very well on tesla, 
            a possibly different gamma_d_nonself value does very well on iedb, 
            both in the context of [key=gamma_logKd_nonself, self_key=SELF_params]
            - this is achieved with the .max() - 
            Additionally, rankings are improved if all gamma_d_nonself values  
            do well for both tesla and iedb
            - this is achieved with the .mean() -
            
            '''
            rho_tesla_contribution = df['tesla_auc'].max() * df['tesla_auc'].mean()
            rho_koncz_contribution = df['koncz_auc'].max() * df['koncz_auc'].mean()
            # ranking_statistic = rho_tesla_contribution *  rho_koncz_contribution
            ranking_statistic = max((rho_tesla_contribution-0.5),0) *  max((rho_koncz_contribution-0.5),0)
            
            # # COMBAT OVERFITTING DUE TO THE FOREIGN_params
            '''
            Add contribution to ranking statistic from model with reduced rho.
            FOREIGN_params==(1e-100,1e-100)        
            (rho is basically 1 in this setting)
            reduced_rho_foreign_list = (gamma_d_foreign, tesla_auc, koncz_auc)
            '''
            reduced_rho_foreign_list = sorted(auc_dict[1e-100][self_params_key], key=lambda x: x[0])[0]
            '''
            Add contribution to ranking statistic from model with reduced rho.
            FOREIGN_params==(gamma_d_foreign,1e-100)         
            (rho depends only on the presentation of the target epitopes under
              the hla of the query)
            '''            
            # reduced_rho_foreign_list = sorted(auc_dict[foreign_gamma_logKd][self_key], key=lambda x: x[0])[0]
            
            reduced_rho_tesla_contribution = reduced_rho_foreign_list[1]
            reduced_rho_koncz_contribution = reduced_rho_foreign_list[2]
            # ranking_statistic *= reduced_rho_tesla_contribution * reduced_rho_koncz_contribution
            ranking_statistic *= max((reduced_rho_tesla_contribution-0.5),0) *  max((reduced_rho_koncz_contribution-0.5),0)
            
            keys_ranking_list.append([foreign_gamma_logKd,self_params_key,rho_tesla_contribution, rho_koncz_contribution, reduced_rho_tesla_contribution, reduced_rho_koncz_contribution, ranking_statistic])

                
    keys_ranking_list.sort(key=lambda x: x[6], reverse=True)
    print("keys_ranking_list[:10]: ", keys_ranking_list[:10])
    print("")
    print("auc_dict[keys_ranking_list[0][0]][keys_ranking_list[0][1]]: ",auc_dict[keys_ranking_list[0][0]][keys_ranking_list[0][1]])
    print("runtime:", time.time() - start_time)
    
    script = "rust_projects/.../analysis4.py"
    print("script: ",script)
    

