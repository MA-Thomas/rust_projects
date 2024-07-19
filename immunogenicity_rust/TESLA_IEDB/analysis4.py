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
def load_combined_dataframe():
    with open('/Users/marcus/Work_Data/Foreign_Epitopes/combined_Tesla_Koncz_dataframe.pkl','rb') as pk:
        hla_df = pickle.load(pk)
        
    hla_df = hla_df[hla_df['allele'] == args.allele]
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

'''
The incorrect formula for the AUC at chance level under class imbalance is:
AUC_chance = (1 + (n_minority / n_majority)) / 2
'''
# def compute_chance_AUC_under_class_imbalance(a,b):
#     n_majority = max(a,b)
#     n_minority = min(a,b)
#     return (1 + (n_minority/n_majority))/2


# # The parameter sets used in Rust.
gamma_d_self_values = sorted(list(set( list(np.round(create_evenly_spaced_list(1e-6, 1e-4, 8),10)) + list(np.round(create_log_spaced_list(2e-4, 1, 7),4)) )))
gamma_logkd_self_values = sorted(list(set( list(np.round(create_log_spaced_list(1e-2, 1.0, 5),4)) + list(np.round(create_log_spaced_list(5e-3, 0.6, 12),4))  )))

gamma_d_nonself_values = sorted(list(set( list(np.round(create_evenly_spaced_list(1e-6, 1e-4, 8),10)) + list(np.round(create_log_spaced_list(1e-1, 5, 6),4)) + [1e-8, 1e-100])))
gamma_logkd_nonself_values = sorted(list(set( list(np.round(create_log_spaced_list(1e-2, 1.0, 5),4)) + list(np.round(create_log_spaced_list(5e-3, 0.6, 10),4))  +[1e-8, 1e-100])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-pkl_dir", default="/Users/marcus/Work_Data/rust_outputs_local/immunogenicity_outputs/d_ub_100_d_lb_0")
    parser.add_argument("-distance_metric_type", default="all_tcr_all_combos_model")
    parser.add_argument("-tesla_variables_dir", default="/Users/marcus/Work_Data/Minerva_editing/CFIT_Editing/bin/TESLA")
    parser.add_argument("-iedb_variables_dir", default="/Users/marcus/Work_Data/Minerva_editing/CFIT_Editing/bin/IEDB")
    parser.add_argument("-allele", default='A0301')
    args = parser.parse_args()

    start_time = time.time()
    # hla_df = load_and_preprocess_data(args)
    hla_df = load_combined_dataframe()
    
    ##%% GATHER PATHS TO PICKLE FILES STORING IMMUNOGENICITY INFO
    tesla_pos_pkl_list = []
    tesla_neg_pkl_list = []
    koncz_pos_pkl_list = []
    koncz_neg_pkl_list = []
    tesla_epi_presentation_tuples_pos = [] 
    tesla_epi_presentation_tuples_neg = [] 
    koncz_epi_presentation_tuples_pos = []
    koncz_epi_presentation_tuples_neg = [] 
    
    for idx, (df_index,row) in enumerate(hla_df.iterrows()):
        if row['assay'] == 'TESLA' and row['immunogenicity']==1:
            query_epi_list = row['peptides']
            full_query_epi = full_pep_from_ninemers(query_epi_list)
            strg = f'immunogenicity_outputs_{args.distance_metric_type}_{idx}_' + '_'.join(query_epi_list) + f'_HLA-{args.allele}.pkl'
            tesla_pos_pkl_list.append( os.path.join(args.pkl_dir, args.allele, strg))
            
            for pep_idx, ninemer in enumerate(query_epi_list):
                # tesla_epi_presentation_tuples_pos.append( (ninemer, 1/np.log(row['Kd_orig_epi']) ) )
                # tesla_epi_presentation_tuples_pos.append( (ninemer, 1/np.log(row['Kds'][pep_idx]) ) )
                tesla_epi_presentation_tuples_pos.append( (ninemer, 1/row['Kds'][pep_idx] ) )
        elif row['assay'] == 'TESLA' and row['immunogenicity']==0:
            query_epi_list = row['peptides']
            full_query_epi = full_pep_from_ninemers(query_epi_list)
            strg = f'immunogenicity_outputs_{args.distance_metric_type}_{idx}_' + '_'.join(query_epi_list) + f'_HLA-{args.allele}.pkl'
            tesla_neg_pkl_list.append( os.path.join(args.pkl_dir, args.allele, strg))
            
            for pep_idx, ninemer in enumerate(query_epi_list):
                # tesla_epi_presentation_tuples_neg.append( (ninemer, 1/np.log(row['Kd_orig_epi']) ) )
                # tesla_epi_presentation_tuples_neg.append( (ninemer, 1/np.log(row['Kds'][pep_idx]) ) )
                tesla_epi_presentation_tuples_neg.append( (ninemer, 1/row['Kds'][pep_idx] ) )
            
        elif row['assay'] == 'IEDB' and row['immunogenicity']==1:
            query_epi_list = row['peptides']
            full_query_epi = full_pep_from_ninemers(query_epi_list)
            strg = f'immunogenicity_outputs_{args.distance_metric_type}_{idx}_' + '_'.join(query_epi_list) + f'_HLA-{args.allele}.pkl'
            koncz_pos_pkl_list.append( os.path.join(args.pkl_dir, args.allele, strg))
           
            for pep_idx, ninemer in enumerate(query_epi_list):
                # koncz_epi_presentation_tuples_pos.append( (ninemer, 1/np.log(row['Kd_orig_epi']) ) )
                # koncz_epi_presentation_tuples_pos.append( (ninemer, 1/np.log(row['Kds'][pep_idx]) ) )
                koncz_epi_presentation_tuples_pos.append( (ninemer, 1/row['Kds'][pep_idx] ) )
            
        elif row['assay'] == 'IEDB' and row['immunogenicity']==0:
            query_epi_list = row['peptides']
            full_query_epi = full_pep_from_ninemers(query_epi_list)
            strg = f'immunogenicity_outputs_{args.distance_metric_type}_{idx}_' + '_'.join(query_epi_list) + f'_HLA-{args.allele}.pkl'
            koncz_neg_pkl_list.append( os.path.join(args.pkl_dir, args.allele, strg))
            
            for pep_idx, ninemer in enumerate(query_epi_list):
                # koncz_epi_presentation_tuples_neg.append( (ninemer, 1/np.log(row['Kd_orig_epi']) ) )
                # koncz_epi_presentation_tuples_neg.append( (ninemer, 1/np.log(row['Kds'][pep_idx]) ) )
                koncz_epi_presentation_tuples_neg.append( (ninemer, 1/row['Kds'][pep_idx] ) )

    # chance_ROC_AUC_under_tesla_class_imbalance = compute_chance_AUC_under_class_imbalance(len(tesla_pos_pkl_list),len(tesla_neg_pkl_list))
    # chance_ROC_AUC_under_koncz_class_imbalance = compute_chance_AUC_under_class_imbalance(len(koncz_pos_pkl_list),len(koncz_neg_pkl_list))
    
    chance_ROC_AUC_under_tesla_class_imbalance = 0.5
    chance_ROC_AUC_under_koncz_class_imbalance = 0.5
    print("chance_ROC_AUC_under_tesla_class_imbalance: ",chance_ROC_AUC_under_tesla_class_imbalance)
    print("chance_ROC_AUC_under_koncz_class_imbalance: ",chance_ROC_AUC_under_koncz_class_imbalance)
    print("len(tesla_pos_pkl_list),len(tesla_neg_pkl_list): ",len(tesla_pos_pkl_list),len(tesla_neg_pkl_list))
    print("len(koncz_pos_pkl_list),len(koncz_neg_pkl_list): ",len(koncz_pos_pkl_list),len(koncz_neg_pkl_list))
###############################################################################
##%% COMPUTE IMM_DICT (rust) AND AUC_DICT (python) -> ~40 seconds

    
    rst = time.time()
    auc_dict = {}
    tesla_pos_imm_dict = immunogenicity_rust.get_immunogenicity_dict_py(tesla_pos_pkl_list) 
    tesla_neg_imm_dict = immunogenicity_rust.get_immunogenicity_dict_py(tesla_neg_pkl_list)

    
    koncz_pos_imm_dict = immunogenicity_rust.get_immunogenicity_dict_py(koncz_pos_pkl_list)
    koncz_neg_imm_dict = immunogenicity_rust.get_immunogenicity_dict_py(koncz_neg_pkl_list)

    print("Time to generate/load immunogenicity_dicts from rust: ", time.time()-rst)
    assert(1==2)
    
    # import re
    # def remove_leading_zeros_scientific_notation(input_str):
    #     if 'e' in input_str:
    #         # Use regular expression to find and remove leading zeros after 'e'
    #         return re.sub(r'e-0+', 'e-', input_str)
    #     else:
    #         return input_str  # Return input_str unchanged if 'e' is not present


    # for gamma_d_self in gamma_d_self_values:
    #     for gamma_logKd_self in gamma_logkd_self_values:
    #         rst = time.time()
    #         auc_dict = {}
            
    #         p1 = remove_leading_zeros_scientific_notation(str(gamma_d_self)) 
    #         p2 = remove_leading_zeros_scientific_notation(str(gamma_logKd_self))
    #         tesla_pos_imm_dict = immunogenicity_rust.get_immunogenicity_dict_py(tesla_pos_pkl_list, (), p1, p2) 
    #         tesla_neg_imm_dict = immunogenicity_rust.get_immunogenicity_dict_py(tesla_neg_pkl_list, (), p1, p2)
        
            
    #         koncz_pos_imm_dict = immunogenicity_rust.get_immunogenicity_dict_py(koncz_pos_pkl_list, (), p1, p2)
    #         koncz_neg_imm_dict = immunogenicity_rust.get_immunogenicity_dict_py(koncz_neg_pkl_list, (), p1, p2)    
    # assert(1==2)
    
    
    
    use_python_auc_function = True
    use_rust_auc_function = True
    
    st = time.time()
    for foreign_key in tesla_pos_imm_dict:
        FOREIGN_params_numeric = string_to_numeric_tuple(foreign_key)

        for self_key in tesla_pos_imm_dict[foreign_key]:
            SELF_params_numeric = string_to_numeric_tuple(self_key)
            
            
            if use_python_auc_function:
                tesla_roc_auc = calculate_auc(tesla_pos_imm_dict[foreign_key][self_key], tesla_neg_imm_dict[foreign_key][self_key])
                koncz_roc_auc = calculate_auc(koncz_pos_imm_dict[foreign_key][self_key], koncz_neg_imm_dict[foreign_key][self_key])
            if use_rust_auc_function:    
                tesla_roc_auc_rust = immunogenicity_rust.calculate_auc_py(tesla_pos_imm_dict[foreign_key][self_key], tesla_neg_imm_dict[foreign_key][self_key])
                koncz_roc_auc_rust = immunogenicity_rust.calculate_auc_py(koncz_pos_imm_dict[foreign_key][self_key], koncz_neg_imm_dict[foreign_key][self_key])
                assert(tesla_roc_auc == round(tesla_roc_auc_rust,12))
                assert(koncz_roc_auc == round(koncz_roc_auc_rust,12))
            
            if FOREIGN_params_numeric[1] not in auc_dict:
                auc_dict[FOREIGN_params_numeric[1]] = {}
            if SELF_params_numeric not in auc_dict[FOREIGN_params_numeric[1]]:
                auc_dict[FOREIGN_params_numeric[1]][SELF_params_numeric] = []
                
            auc_dict[FOREIGN_params_numeric[1]][SELF_params_numeric].append([FOREIGN_params_numeric[0], tesla_roc_auc, koncz_roc_auc])
   
    print("runtime to eval auc_dict in python: ", time.time()-st)
    assert(1==4)


###############################################################################
#%% COMPUTE AUC_DICT DIRECTLY IN RUST

    use_tesla = True 
    use_Koncz = True
    use_query_presentation = False
    
    if not use_query_presentation:
        tesla_epi_presentation_tuples_pos = None
        tesla_epi_presentation_tuples_neg = None 
        koncz_epi_presentation_tuples_pos = None 
        koncz_epi_presentation_tuples_neg = None 
        
    num_self_params_per_iter = 30
    if len(tesla_pos_pkl_list) > 0 and len(tesla_neg_pkl_list) > 0:
        auc_type_str = "ROC" 
        tesla_roc_auc_dict = immunogenicity_rust.calculate_auc_dict_from_pickle_files_py(tesla_pos_pkl_list,tesla_neg_pkl_list, num_self_params_per_iter, auc_type_str, tesla_epi_presentation_tuples_pos, tesla_epi_presentation_tuples_neg)
        auc_type_str = "PR" 
        tesla_pr_auc_dict = immunogenicity_rust.calculate_auc_dict_from_pickle_files_py(tesla_pos_pkl_list,tesla_neg_pkl_list, num_self_params_per_iter, auc_type_str, tesla_epi_presentation_tuples_pos, tesla_epi_presentation_tuples_neg)
        print("len(tesla_roc_auc_dict): ", len(tesla_roc_auc_dict))
    else:
        use_tesla = False
        
    if len(koncz_pos_pkl_list) > 0 and len(koncz_neg_pkl_list) > 0:
        auc_type_str = "ROC" 
        koncz_roc_auc_dict = immunogenicity_rust.calculate_auc_dict_from_pickle_files_py(koncz_pos_pkl_list,koncz_neg_pkl_list, num_self_params_per_iter, auc_type_str, koncz_epi_presentation_tuples_pos, koncz_epi_presentation_tuples_neg)
        auc_type_str = "PR" 
        koncz_pr_auc_dict = immunogenicity_rust.calculate_auc_dict_from_pickle_files_py(koncz_pos_pkl_list,koncz_neg_pkl_list, num_self_params_per_iter, auc_type_str, koncz_epi_presentation_tuples_pos, koncz_epi_presentation_tuples_neg)
        print("len(koncz_roc_auc_dict): ", len(koncz_roc_auc_dict))
    else:
        use_Koncz = False

    
    # # We need a dict to iterate over parameters with.
    # # If tesla_roc_auc_dict doesn't exist, use koncz_roc_auc_dict (and vice versa)
    if use_tesla == False:
        dict_structure = koncz_roc_auc_dict 
    else: 
        dict_structure = tesla_roc_auc_dict 
        
#%% COMBINE AUC_DICTS AND UPDATE STRUCTURE
    foreign_key_1_set = set()
    foreign_key_0_set = set()
    auc_dict = {}
    for self_key_str in dict_structure:
        self_key = ast.literal_eval(self_key_str)
        
        for foreign_key_str in dict_structure[self_key_str]:
            foreign_key = ast.literal_eval(foreign_key_str)
            
            
            if foreign_key[1] not in auc_dict:
                print("foreign_key[1]: ",foreign_key[1])
                auc_dict[foreign_key[1]] = {}
                
            if self_key not in auc_dict[foreign_key[1]]: 
                auc_dict[foreign_key[1]][self_key] = []

            if len(tesla_pos_pkl_list) > 0 and len(tesla_neg_pkl_list) > 0:
                tesla_roc_auc = tesla_roc_auc_dict[self_key_str][foreign_key_str]
                tesla_pr_auc = tesla_pr_auc_dict[self_key_str][foreign_key_str]
            else:
                tesla_roc_auc = -1
                tesla_pr_auc
            
            if len(koncz_pos_pkl_list) > 0 and len(koncz_neg_pkl_list) > 0:
                koncz_roc_auc = koncz_roc_auc_dict[self_key_str][foreign_key_str]
                koncz_pr_auc = koncz_pr_auc_dict[self_key_str][foreign_key_str]
            else:
                koncz_roc_auc = -1
                koncz_pr_auc = -1
            
            auc_dict[foreign_key[1]][self_key].append( (foreign_key[0],tesla_roc_auc, koncz_roc_auc, tesla_pr_auc, koncz_pr_auc ) )
            

#%%
    use_full_rho_in_ranking = True
    use_reduced_rho_in_ranking = True
    
    keys_ranking_list = []
    for gamma_logKd_foreign in auc_dict:
        for self_params_key in auc_dict[gamma_logKd_foreign]:
            
            auc_dict[gamma_logKd_foreign][self_params_key].sort(key=lambda x: x[1], reverse=True)

            # # Convert the list to a DataFrame
            df = pd.DataFrame(auc_dict[gamma_logKd_foreign][self_params_key], columns=['gamma_d_nonself', 'tesla_roc_auc', 'koncz_roc_auc', 'tesla_pr_auc', 'koncz_pr_auc'])
            
            
            '''
            Provide a ranking for parameter triplet [gamma_logKd_foreign, SELF_params].
            A higher ranking implies that:  
            a particular gamma_d_nonself value does very well on tesla, 
            a possibly different gamma_d_nonself value does very well on iedb, 
            both in the context of [key=gamma_logKd_nonself, self_key=SELF_params]
            - this is achieved with the .max() - 
            Additionally, rankings are improved if all gamma_d_nonself values  
            do well for both tesla and iedb
            - this is achieved with the .mean() -
            
            '''
            rho_tesla_roc_contribution = df['tesla_roc_auc'].max() #* df['tesla_roc_auc'].mean()
            rho_koncz_roc_contribution = df['koncz_roc_auc'].max() #* df['koncz_roc_auc'].mean()
            if use_full_rho_in_ranking:
                # ranking_statistic = (rho_tesla_roc_contribution / chance_ROC_AUC_under_tesla_class_imbalance) * (rho_koncz_roc_contribution / chance_ROC_AUC_under_koncz_class_imbalance)
                ranking_statistic = (rho_tesla_roc_contribution - chance_ROC_AUC_under_tesla_class_imbalance) * (rho_koncz_roc_contribution - chance_ROC_AUC_under_koncz_class_imbalance)

                # ranking_statistic = rho_tesla_roc_contribution * rho_koncz_roc_contribution

            else: 
                ranking_statistic = 1
            
            # # Ranking is based only on roc auc, not pr auc
            rho_tesla_pr_contribution = df['tesla_pr_auc'].max() #* df['tesla_pr_auc'].mean()
            rho_koncz_pr_contribution = df['koncz_pr_auc'].max() #* df['koncz_pr_auc'].mean()
            
            
            
            
            
            
            '''
            COMBAT OVERFITTING DUE TO THE FOREIGN_params
            Add contribution to ranking statistic from model with reduced rho.
            FOREIGN_params==(1e-100,1e-100)        
            (rho is basically 1 in this setting)
            reduced_rho_foreign_list = (gamma_d_foreign, tesla_roc_auc, koncz_roc_auc)
            '''
            reduced_rho_foreign_list = sorted(auc_dict[1e-100][self_params_key], key=lambda x: x[0])[0]
            
            reduced_rho_tesla_roc_contribution = reduced_rho_foreign_list[1] 
            reduced_rho_koncz_roc_contribution = reduced_rho_foreign_list[2] 
            if use_reduced_rho_in_ranking:
                ranking_statistic *= (reduced_rho_tesla_roc_contribution - chance_ROC_AUC_under_tesla_class_imbalance) * (reduced_rho_koncz_roc_contribution - chance_ROC_AUC_under_koncz_class_imbalance)

            else:
                ranking_statistic *= 1
            
            # # Ranking is based only on roc auc, not pr auc
            reduced_rho_tesla_pr_contribution = reduced_rho_foreign_list[3]
            reduced_rho_koncz_pr_contribution = reduced_rho_foreign_list[4]
            
            


            keys_ranking_list.append([gamma_logKd_foreign,self_params_key,rho_tesla_roc_contribution, rho_koncz_roc_contribution, reduced_rho_tesla_roc_contribution, reduced_rho_koncz_roc_contribution, rho_tesla_pr_contribution, rho_koncz_pr_contribution, reduced_rho_tesla_pr_contribution, reduced_rho_koncz_pr_contribution, ranking_statistic])

                
    keys_ranking_list.sort(key=lambda x: x[10], reverse=True)
    print("keys_ranking_list[:10]: ", keys_ranking_list[:10])
    print("")
    print("auc_dict[keys_ranking_list[0][0]][keys_ranking_list[0][1]]: ",auc_dict[keys_ranking_list[0][0]][keys_ranking_list[0][1]])
    print("runtime:", time.time() - start_time)
    
    script = "rust_projects/.../analysis4.py"
    print("script: ",script)
    
    keys_ranking_df = pd.DataFrame(keys_ranking_list, columns=['gamma_d_logKd_foreign', 'self_params','tesla_roc_contribution', 'koncz_roc_contribution', 'reduced_rho_tesla_roc_contribution', 'reduced_rho_koncz_roc_contribution', 'tesla_pr_contribution', 'koncz_pr_contribution', 'reduced_rho_tesla_pr_contribution', 'reduced_rho_koncz_pr_contribution', 'ranking_statistic'])
    
    print("len(tesla_pos_pkl_list),len(tesla_neg_pkl_list): ",len(tesla_pos_pkl_list),len(tesla_neg_pkl_list))
    print("len(koncz_pos_pkl_list),len(koncz_neg_pkl_list): ",len(koncz_pos_pkl_list),len(koncz_neg_pkl_list))
